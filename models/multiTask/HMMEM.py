import math
import os
import sys
import collections
from torch.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.subNets.Textmodel import Language_model
from .HMMEM_loss import DiffLoss, LightweightCrossCPC
from .HMMEM_moe import GlobalMoE, LocalMoE
from .HMMEM_modules import TVA_LSTM, Text_guide_mixer, Lightweight_mixer, mutli_scale_fusion, FeatureAdapter
from .HMMEM_mixer import AdaptiveModalMixer

__all__ = ['HMMEM']


class HMMEM(nn.Module):
    def __init__(self, args):
        super(HMMEM, self).__init__()
        # text encoding
        self.LLM = Language_model(args)

        # audio and video encoding
        text_in, audio_in, video_in = args.feature_dims[:]
        text_len, audio_len, video_len = args.seq_lens[:]

        # Dynamically resolve text_in from LLM hidden_size
        if hasattr(self.LLM.model, 'config'):
            config = self.LLM.model.config
            llm_hidden_size = getattr(config, 'hidden_size', None)
            # Fallback: multimodal models (e.g. Gemma 4) nest hidden_size under text_config
            if llm_hidden_size is None and hasattr(config, 'text_config'):
                llm_hidden_size = getattr(config.text_config, 'hidden_size', None)
            # Fallback: detect from embedding layer output dimension
            if llm_hidden_size is None:
                embed_layer = self.LLM.model.get_input_embeddings()
                if embed_layer is not None and hasattr(embed_layer, 'embedding_dim'):
                    llm_hidden_size = embed_layer.embedding_dim
            if llm_hidden_size is not None and (text_in == 0 or text_in != llm_hidden_size):
                text_in = llm_hidden_size
                args.feature_dims = (text_in, audio_in, video_in)

        # ── Feature Adapter (for high-dim encoders like HuBERT/Whisper) ──
        adapter_dim = getattr(args, 'adapter_dim', 128)
        if audio_in > adapter_dim:
            self.audio_adapter = FeatureAdapter(audio_in, adapter_dim)
            lstm_audio_in = adapter_dim
        else:
            self.audio_adapter = None
            lstm_audio_in = audio_in

        if video_in > adapter_dim:
            self.video_adapter = FeatureAdapter(video_in, adapter_dim)
            lstm_video_in = adapter_dim
        else:
            self.video_adapter = None
            lstm_video_in = video_in

        self.audio_LSTM = TVA_LSTM(lstm_audio_in, args.a_lstm_hidden_size, num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_LSTM = TVA_LSTM(lstm_video_in, args.v_lstm_hidden_size, num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # ── Feature flags ──
        # Mixer layer (mutually exclusive: use_amm overrides use_tgm)
        self.use_tgm = getattr(args, 'use_tgm', True)
        self.use_amm = getattr(args, 'use_amm', False)
        if self.use_amm:
            self.use_tgm = False  # AMM overrides TGM

        # Fusion layer (mutually exclusive: use_moe_fusion overrides use_msf)
        self.use_msf = getattr(args, 'use_msf', True)
        self.use_moe_fusion = getattr(args, 'use_moe_fusion', False)
        if self.use_moe_fusion:
            self.use_msf = False  # MoE overrides MSF

        # Auxiliary losses
        self.use_gate = getattr(args, 'use_gate', False)
        self.use_moe_lb_loss = getattr(args, 'use_moe_lb_loss', False)
        self.use_diff_loss = getattr(args, 'use_diff_loss', False)
        self.use_expert_diff_loss = getattr(args, 'use_expert_diff_loss', False)
        self.use_nce_loss = getattr(args, 'use_nce_loss', False)
        self.diff_loss_weight = getattr(args, 'diff_loss_weight', 0.01)
        self.nce_weight = getattr(args, 'nce_weight', 0.05)

        fusion_input_size = 256
        self.text_in = text_in

        # ══════════════════════════════════════════════
        # Mixer Layer
        # ══════════════════════════════════════════════
        if self.use_amm:
            self.mixer = AdaptiveModalMixer(text_in=text_in, fusion_dim=fusion_input_size)
        elif self.use_tgm:
            self.mixer = Text_guide_mixer(text_in)
        else:
            # Fallback: simple audio + video (no text guidance)
            self.mixer = Lightweight_mixer()

        # ══════════════════════════════════════════════
        # Fusion Layer
        # ══════════════════════════════════════════════
        if self.use_moe_fusion:
            # ── Dual-Branch MoE ──
            self.pseudo_tokens = args.pseudo_tokens
            num_local = getattr(args, 'num_local_experts', 3)
            expert_bottleneck = getattr(args, 'expert_bottleneck', 64)

            # Global Emotion MoE (heterogeneous experts)
            self.global_moe = GlobalMoE(input_size=fusion_input_size)

            # Local Emotion MoE (bottleneck experts, receives raw modality features)
            self.local_moe = LocalMoE(
                input_size=fusion_input_size,
                num_experts=num_local,
                bottleneck_dim=expert_bottleneck
            )
            # Projection for Local branch input: cat(audio_h, video_h) [512] → [256]
            self.local_proj = nn.Linear(fusion_input_size * 2, fusion_input_size)

            # Meta-Gate: fuse Global and Local branches
            # Input: feature_f (256) + (global_out - local_out) (256) + optional cosine bias (1)
            meta_gate_input_dim = fusion_input_size * 2  # 512
            if self.use_gate:
                meta_gate_input_dim += 1  # + cosine bias
            self.meta_gate = nn.Linear(meta_gate_input_dim, 2)

            # Shared projector: 256 → text_in → pseudo_tokens
            self.shared_projector = nn.Linear(fusion_input_size, text_in)
            self.shared_token_projector = nn.Linear(1, args.pseudo_tokens)

        elif self.use_msf:
            # ── Original multi_scale_fusion (baseline) ──
            self.fusion = mutli_scale_fusion(
                input_size=fusion_input_size,
                output_size=text_in,
                pseudo_tokens=args.pseudo_tokens
            )
        else:
            # ── Direct projection fallback ──
            self.direct_proj = nn.Linear(fusion_input_size, text_in)
            self.direct_token_proj = nn.Linear(1, args.pseudo_tokens)

        # ── Optional: DiffLoss ──
        if self.use_diff_loss or self.use_expert_diff_loss:
            self.diff_loss_fn = DiffLoss()

        # ── Optional: NCE Loss ──
        if self.use_nce_loss:
            nce_hidden_dim = getattr(args, 'nce_hidden_dim', 32)
            nce_pred_steps = getattr(args, 'nce_pred_steps', 2)
            self.cpc_text_audio = LightweightCrossCPC(
                text_dim=text_in, other_dim=args.a_lstm_hidden_size,
                nce_hidden_dim=nce_hidden_dim, n_prediction_steps=nce_pred_steps
            )
            self.cpc_text_video = LightweightCrossCPC(
                text_dim=text_in, other_dim=args.v_lstm_hidden_size,
                nce_hidden_dim=nce_hidden_dim, n_prediction_steps=nce_pred_steps
            )

    # ──────────────────────────────────────────────
    # Helper methods
    # ──────────────────────────────────────────────

    def _compute_lb_loss(self, gate_weights):
        """Entropy-based load-balance loss for N≥3 homogeneous experts."""
        expert_load = torch.mean(gate_weights, dim=0)
        lb_loss = -torch.sum(expert_load * torch.log(expert_load + 1e-10))
        return lb_loss

    def _compute_diff_loss_pairs(self, expert_outputs):
        """Pairwise DiffLoss between all expert outputs."""
        n = len(expert_outputs)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total = total + self.diff_loss_fn(expert_outputs[i], expert_outputs[j])
                count += 1
        return total / max(count, 1)

    def _dual_moe_forward(self, audio_h, video_h, feature_f):
        """Dual-Branch MoE forward: Global MoE + Local MoE + Meta-Gate."""
        # 1. Global MoE — receives fused feature_f
        global_out, global_gw, global_experts = self.global_moe(feature_f)

        # 2. Local MoE — receives raw modality features (differentiated input!)
        local_input = self.local_proj(torch.cat([audio_h, video_h], dim=-1))
        local_out, local_gw, local_experts = self.local_moe(local_input)

        # 3. Meta-Gate (posterior-aware: sees branch output difference)
        meta_input = torch.cat([feature_f, global_out - local_out], dim=-1)  # [B, 512]
        if self.use_gate:
            bias_score = F.cosine_similarity(audio_h, video_h, dim=-1).unsqueeze(-1)
            meta_input = torch.cat([meta_input, bias_score], dim=-1)
        meta_weights = F.softmax(self.meta_gate(meta_input), dim=-1)  # [B, 2]

        fused = meta_weights[:, 0:1] * global_out + meta_weights[:, 1:2] * local_out

        # 4. Shared projector → [B, pseudo_tokens, text_in]
        projected = self.shared_projector(fused)
        fusion_h = self.shared_token_projector(projected.unsqueeze(2))
        fusion_h = fusion_h.permute(0, 2, 1)

        aux = {
            'global_gw': global_gw,
            'local_gw': local_gw,
            'meta_weights': meta_weights,
            'global_out': global_out,
            'local_out': local_out,
            'global_experts': global_experts,
            'local_experts': local_experts,
        }
        return fusion_h, aux

    def _apply_fusion(self, feature_f):
        """Apply the non-MoE fusion path (MSF or direct projection)."""
        if self.use_msf:
            return self.fusion(feature_f)
        else:
            projected = self.direct_proj(feature_f)
            fusion_h = self.direct_token_proj(projected.unsqueeze(2))
            return fusion_h.permute(0, 2, 1)

    # ──────────────────────────────────────────────
    # Forward / Generate
    # ──────────────────────────────────────────────

    def forward(self, labels, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text

        # Feature Adapter (if present)
        if self.audio_adapter is not None:
            audio = self.audio_adapter(audio)
        if self.video_adapter is not None:
            video = self.video_adapter(video)

        # Text embedding
        text_embed = self.LLM.text_embedding(text[:, 0, :].long())

        # Audio & Video encoding
        if self.use_nce_loss:
            audio_h, audio_seq = self.audio_LSTM(audio, audio_len, return_sequence=True)
            video_h, video_seq = self.video_LSTM(video, video_len, return_sequence=True)
        else:
            audio_h = self.audio_LSTM(audio, audio_len)
            video_h = self.video_LSTM(video, video_len)

        # Mixer
        feature_f = self.mixer(audio_h, video_h, text_embed)

        # Fusion
        if self.use_moe_fusion:
            fusion_h, moe_aux = self._dual_moe_forward(audio_h, video_h, feature_f)
        else:
            fusion_h = self._apply_fusion(feature_f)

        # LLM forward
        LLM_input = torch.cat([fusion_h, text_embed], dim=1)
        LLM_output = self.LLM(LLM_input, labels)

        res = {
            'Loss': LLM_output.loss,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': feature_f,
        }

        # ── Auxiliary Losses (training only) ──
        if self.use_moe_fusion and self.training:
            if self.use_moe_lb_loss:
                # Only apply LB loss to Local branch (Global has heterogeneous experts)
                lb_local = self._compute_lb_loss(moe_aux['local_gw'])
                res['MoE_LB_Loss'] = lb_local * 0.01

            if self.use_diff_loss:
                diff_branch = self.diff_loss_fn(moe_aux['global_out'], moe_aux['local_out'])
                res['DiffLoss'] = diff_branch * self.diff_loss_weight

            if self.use_expert_diff_loss:
                diff_global = self._compute_diff_loss_pairs(moe_aux['global_experts'])
                diff_local = self._compute_diff_loss_pairs(moe_aux['local_experts'])
                res['ExpertDiffLoss'] = (diff_global + diff_local) * self.diff_loss_weight

        if self.use_nce_loss and self.training:
            nce_ta = self.cpc_text_audio(text_embed, audio_seq)
            nce_tv = self.cpc_text_video(text_embed, video_seq)
            res['NCELoss'] = (nce_ta + nce_tv) * self.nce_weight

        return res

    def generate(self, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text

        # Feature Adapter
        if self.audio_adapter is not None:
            audio = self.audio_adapter(audio)
        if self.video_adapter is not None:
            video = self.video_adapter(video)

        text_embed = self.LLM.text_embedding(text[:, 0, :].long())

        audio_h = self.audio_LSTM(audio, audio_len)
        video_h = self.video_LSTM(video, video_len)

        # Mixer
        feature_f = self.mixer(audio_h, video_h, text_embed)

        # Fusion
        if self.use_moe_fusion:
            fusion_h, _ = self._dual_moe_forward(audio_h, video_h, feature_f)
        else:
            fusion_h = self._apply_fusion(feature_f)

        LLM_input = torch.cat([fusion_h, text_embed], dim=1)
        LLM_output = self.LLM.generate(LLM_input)

        return LLM_output
