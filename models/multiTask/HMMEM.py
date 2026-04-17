import math
import os
import sys
import collections
from torch.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_modules import Language_model
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
        # Each modality can have its own adapter dim; falls back to shared adapter_dim
        _adapter_dim = getattr(args, 'adapter_dim', 128)
        audio_adapter_dim = getattr(args, 'audio_adapter_dim', None) or _adapter_dim
        video_adapter_dim = getattr(args, 'video_adapter_dim', None) or _adapter_dim

        if audio_in > audio_adapter_dim:
            self.audio_adapter = FeatureAdapter(audio_in, audio_adapter_dim)
            lstm_audio_in = audio_adapter_dim
        else:
            self.audio_adapter = None
            lstm_audio_in = audio_in

        if video_in > video_adapter_dim:
            self.video_adapter = FeatureAdapter(video_in, video_adapter_dim)
            lstm_video_in = video_adapter_dim
        else:
            self.video_adapter = None
            lstm_video_in = video_in

        self.audio_LSTM = TVA_LSTM(lstm_audio_in, args.a_lstm_hidden_size, num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_LSTM = TVA_LSTM(lstm_video_in, args.v_lstm_hidden_size, num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # ══════════════════════════════════════════════
        # Modality Ablation: parse --modalities flag
        # ══════════════════════════════════════════════
        modalities = str(getattr(args, 'modalities', 'tav')).lower()
        self.use_text  = 't' in modalities
        self.use_audio = 'a' in modalities
        self.use_video = 'v' in modalities
        self.text_only = self.use_text and not self.use_audio and not self.use_video
        self.has_av    = self.use_audio or self.use_video  # any non-text modality active
        if not (self.use_text or self.use_audio or self.use_video):
            raise ValueError("--modalities must contain at least one of: t, a, v")

        # ── Feature flags ──
        # Mixer layer (mutually exclusive: use_amm overrides use_tgm)
        # Mixers require text + at least one AV modality
        self.use_tgm = getattr(args, 'use_tgm', True) and self.use_text and self.has_av
        self.use_amm = getattr(args, 'use_amm', False) and self.use_text and self.has_av
        if self.use_amm:
            self.use_tgm = False  # AMM overrides TGM

        # Fusion layer (mutually exclusive: use_moe_fusion overrides use_msf)
        # Fusion only makes sense when AV modalities are present
        self.use_msf = getattr(args, 'use_msf', True) and self.has_av
        self.use_moe_fusion = getattr(args, 'use_moe_fusion', False) and self.has_av
        if self.use_moe_fusion:
            self.use_msf = False  # MoE overrides MSF

        # Auxiliary losses (only meaningful with AV)
        self.use_gate = getattr(args, 'use_gate', False) and self.has_av
        self.use_moe_lb_loss = getattr(args, 'use_moe_lb_loss', False) and self.has_av
        self.use_diff_loss = getattr(args, 'use_diff_loss', False) and self.has_av
        self.use_expert_diff_loss = getattr(args, 'use_expert_diff_loss', False) and self.has_av
        self.use_nce_loss = getattr(args, 'use_nce_loss', False) and self.use_text and self.has_av
        self.diff_loss_weight = getattr(args, 'diff_loss_weight', 0.01)
        self.nce_weight = getattr(args, 'nce_weight', 0.05)

        fusion_input_size = 256
        self.text_in = text_in

        # ══════════════════════════════════════════════
        # Mixer Layer (only built when needed)
        # ══════════════════════════════════════════════
        if self.use_amm:
            self.mixer = AdaptiveModalMixer(text_in=text_in, fusion_dim=fusion_input_size)
        elif self.use_tgm:
            self.mixer = Text_guide_mixer(text_in)
        elif self.has_av:
            # Fallback: simple audio + video (no text guidance)
            self.mixer = Lightweight_mixer()

        # ══════════════════════════════════════════════
        # Fusion Layer (only built when AV present)
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
        elif self.has_av:
            # ── Direct projection fallback ──
            self.direct_proj = nn.Linear(fusion_input_size, text_in)
            self.direct_token_proj = nn.Linear(1, args.pseudo_tokens)

        # ── text-only fallback: project pooled text → fusion_size → LLM ──
        if not self.has_av:
            self.text_fallback_proj = nn.Sequential(
                nn.Linear(text_in, fusion_input_size),
                nn.GELU(),
                nn.Linear(fusion_input_size, text_in),
            )
            self.text_fallback_token_proj = nn.Linear(1, args.pseudo_tokens)

        # ── Raw AV Token Bypass (EmotionLLaMA-v2 style) ──
        raw_av_mode = getattr(args, 'raw_av_mode', 'none').lower()
        self.inject_audio_tokens = raw_av_mode in ('audio', 'both') and self.use_audio
        self.inject_video_tokens = raw_av_mode in ('video', 'both') and self.use_video
        av_pseudo_tokens = getattr(args, 'av_pseudo_tokens', 4)
        if self.inject_audio_tokens:
            self.audio_proj = nn.Linear(256, text_in)
            self.audio_token_proj = nn.Linear(1, av_pseudo_tokens)
        if self.inject_video_tokens:
            self.video_proj = nn.Linear(256, text_in)
            self.video_token_proj = nn.Linear(1, av_pseudo_tokens)

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

    def _build_llm_input(self, fusion_h, text_embed, audio_h, video_h, audio_raw, video_raw):
        """Build LLM input sequence with optional raw AV token bypass and dynamic mask.

        When raw_av_mode != 'none', injects separately projected audio/video tokens
        alongside the fused pseudo-tokens. Dynamic attention mask zeros out tokens
        for samples whose raw features are all-zero (missing modality).

        Returns:
            llm_input: [B, total_seq_len, hidden_size]
            input_attn_mask: [B, total_seq_len] or None
        """
        batch_size = fusion_h.shape[0]
        device = fusion_h.device

        components = [fusion_h]
        mask_parts = [torch.ones(batch_size, fusion_h.shape[1], dtype=torch.long, device=device)]

        if self.inject_audio_tokens:
            audio_tokens = self.audio_proj(audio_h)                          # [B, text_in]
            audio_tokens = self.audio_token_proj(audio_tokens.unsqueeze(2))  # [B, text_in, av_pt]
            audio_tokens = audio_tokens.permute(0, 2, 1)                    # [B, av_pt, text_in]
            # Dynamic mask: detect zero raw audio
            audio_zero = (audio_raw.abs().sum(dim=list(range(1, audio_raw.dim()))) == 0)  # [B]
            audio_mask = (~audio_zero).long().unsqueeze(1).expand(-1, audio_tokens.shape[1])
            components.append(audio_tokens)
            mask_parts.append(audio_mask)

        if self.inject_video_tokens:
            video_tokens = self.video_proj(video_h)                          # [B, text_in]
            video_tokens = self.video_token_proj(video_tokens.unsqueeze(2))  # [B, text_in, av_pt]
            video_tokens = video_tokens.permute(0, 2, 1)                    # [B, av_pt, text_in]
            # Dynamic mask: detect zero raw video
            video_zero = (video_raw.abs().sum(dim=list(range(1, video_raw.dim()))) == 0)  # [B]
            video_mask = (~video_zero).long().unsqueeze(1).expand(-1, video_tokens.shape[1])
            components.append(video_tokens)
            mask_parts.append(video_mask)

        components.append(text_embed)
        mask_parts.append(torch.ones(batch_size, text_embed.shape[1], dtype=torch.long, device=device))

        llm_input = torch.cat(components, dim=1)

        if self.inject_audio_tokens or self.inject_video_tokens:
            input_attn_mask = torch.cat(mask_parts, dim=1)
        else:
            input_attn_mask = None

        return llm_input, input_attn_mask

    def _mix_modalities(self, audio_h, video_h, text_embed):
        """Fuse enabled modalities into a single [B, 256] feature vector.
        
        Handles all 7 ablation combinations cleanly:
          tav: mixer(a, v, t)     ta: mixer(a, zero_v, t)   tv: mixer(zero_a, v, t)
          av:  a + v              a:  a                      v:  v
          t:   text_fallback_proj(pool(text))
        """
        if self.use_audio and self.use_video:
            # Both AV present: full mixer
            if self.use_text:
                return self.mixer(audio_h, video_h, text_embed)
            return audio_h + video_h
        elif self.use_audio:
            # Audio only (or ta)
            zero_v = torch.zeros_like(audio_h)
            if self.use_text:
                return self.mixer(audio_h, zero_v, text_embed)
            return audio_h
        elif self.use_video:
            # Video only (or tv)
            zero_a = torch.zeros_like(video_h)
            if self.use_text:
                return self.mixer(zero_a, video_h, text_embed)
            return video_h
        else:
            # Text only: pool text → project to fusion size
            text_pooled = torch.mean(text_embed, dim=1)  # [B, text_in]
            projected = self.text_fallback_proj(text_pooled)  # [B, text_in]
            fusion_h = self.text_fallback_token_proj(projected.unsqueeze(2))  # [B, text_in, pseudo_tokens]
            return fusion_h.permute(0, 2, 1)  # [B, pseudo_tokens, text_in]

    # ──────────────────────────────────────────────
    # Forward / Generate
    # ──────────────────────────────────────────────

    def forward(self, labels, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text
        batch_size = text.size(0)

        # Save raw features for dynamic zero-detection (before adapter)
        audio_raw, video_raw = audio, video

        # Feature Adapter (if present)
        if self.audio_adapter is not None:
            audio = self.audio_adapter(audio)
        if self.video_adapter is not None:
            video = self.video_adapter(video)

        # ── Text embedding (always computed — needed for LLM input) ──
        text_embed = self.LLM.text_embedding(text[:, 0, :].long())
        if not self.use_text:
            text_embed = torch.zeros_like(text_embed)

        # ── Audio & Video encoding (conditional on enabled modalities) ──
        audio_h = text_embed.new_zeros(batch_size, 256)
        video_h = text_embed.new_zeros(batch_size, 256)
        audio_seq, video_seq = None, None

        if self.use_audio:
            if self.use_nce_loss:
                audio_h, audio_seq = self.audio_LSTM(audio, audio_len, return_sequence=True)
            else:
                audio_h = self.audio_LSTM(audio, audio_len)
        if self.use_video:
            if self.use_nce_loss:
                video_h, video_seq = self.video_LSTM(video, video_len, return_sequence=True)
            else:
                video_h = self.video_LSTM(video, video_len)

        # ── Mixer → Fusion ──
        if self.text_only:
            # Text-only: skip mixer/fusion, use dedicated text path
            fusion_h = self._mix_modalities(audio_h, video_h, text_embed)
            feature_f = text_embed.new_zeros(batch_size, 256)
        elif self.use_moe_fusion:
            feature_f = self._mix_modalities(audio_h, video_h, text_embed)
            fusion_h, moe_aux = self._dual_moe_forward(audio_h, video_h, feature_f)
        else:
            feature_f = self._mix_modalities(audio_h, video_h, text_embed)
            fusion_h = self._apply_fusion(feature_f)

        # ── Build LLM input with optional AV token bypass ──
        LLM_input, input_attn_mask = self._build_llm_input(
            fusion_h, text_embed, audio_h, video_h, audio_raw, video_raw)
        LLM_output = self.LLM(LLM_input, labels, input_attn_mask=input_attn_mask)

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
            nce_terms = []
            if self.use_audio and audio_seq is not None:
                nce_terms.append(self.cpc_text_audio(text_embed, audio_seq))
            if self.use_video and video_seq is not None:
                nce_terms.append(self.cpc_text_video(text_embed, video_seq))
            if nce_terms:
                res['NCELoss'] = sum(nce_terms) * self.nce_weight

        return res

    def generate(self, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text
        batch_size = text.size(0)

        # Save raw features for dynamic zero-detection (before adapter)
        audio_raw, video_raw = audio, video

        # Feature Adapter
        if self.audio_adapter is not None:
            audio = self.audio_adapter(audio)
        if self.video_adapter is not None:
            video = self.video_adapter(video)

        text_embed = self.LLM.text_embedding(text[:, 0, :].long())
        if not self.use_text:
            text_embed = torch.zeros_like(text_embed)

        audio_h = text_embed.new_zeros(batch_size, 256)
        video_h = text_embed.new_zeros(batch_size, 256)
        if self.use_audio:
            audio_h = self.audio_LSTM(audio, audio_len)
        if self.use_video:
            video_h = self.video_LSTM(video, video_len)

        # Mixer → Fusion
        if self.text_only:
            fusion_h = self._mix_modalities(audio_h, video_h, text_embed)
        elif self.use_moe_fusion:
            feature_f = self._mix_modalities(audio_h, video_h, text_embed)
            fusion_h, _ = self._dual_moe_forward(audio_h, video_h, feature_f)
        else:
            feature_f = self._mix_modalities(audio_h, video_h, text_embed)
            fusion_h = self._apply_fusion(feature_f)

        # ── Build LLM input with optional AV token bypass ──
        LLM_input, input_attn_mask = self._build_llm_input(
            fusion_h, text_embed, audio_h, video_h, audio_raw, video_raw)
        LLM_output = self.LLM.generate(LLM_input, input_attn_mask=input_attn_mask)

        return LLM_output
