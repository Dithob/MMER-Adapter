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
from .HMMEM_modules import TVA_LSTM, Text_guide_mixer, mutli_scale_fusion

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
            llm_hidden_size = getattr(self.LLM.model.config, 'hidden_size', None)
            if llm_hidden_size is not None and (text_in == 0 or text_in != llm_hidden_size):
                text_in = llm_hidden_size
                args.feature_dims = (text_in, audio_in, video_in)

        self.audio_LSTM = TVA_LSTM(audio_in, args.a_lstm_hidden_size, num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_LSTM = TVA_LSTM(video_in, args.v_lstm_hidden_size, num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # Feature flags
        self.use_moe_fusion = getattr(args, 'use_moe_fusion', False)
        self.use_gate = getattr(args, 'use_gate', False)
        self.use_moe_lb_loss = getattr(args, 'use_moe_lb_loss', False)
        self.use_diff_loss = getattr(args, 'use_diff_loss', False)
        self.use_expert_diff_loss = getattr(args, 'use_expert_diff_loss', False)
        self.use_nce_loss = getattr(args, 'use_nce_loss', False)
        self.diff_loss_weight = getattr(args, 'diff_loss_weight', 0.01)
        self.nce_weight = getattr(args, 'nce_weight', 0.05)

        fusion_input_size = 256
        self.text_in = text_in

        # Shared Text_guide_mixer (used by all paths)
        self.text_guide_mixer = Text_guide_mixer(text_in)

        if self.use_moe_fusion:
            # ── Dual-Branch MoE ──
            self.pseudo_tokens = args.pseudo_tokens
            num_global = getattr(args, 'num_global_experts', 3)
            num_local = getattr(args, 'num_local_experts', 3)
            expert_bottleneck = getattr(args, 'expert_bottleneck', 64)

            # Global Emotion MoE (multi-scale upgrade)
            self.global_moe = GlobalMoE(
                input_size=fusion_input_size,
                output_size=text_in,
                num_experts=num_global
            )
            # Local Emotion MoE (lightweight bottleneck)
            self.local_moe = LocalMoE(
                input_size=fusion_input_size,
                num_experts=num_local,
                bottleneck_dim=expert_bottleneck
            )

            # Meta-Gate: fuse Global and Local branches
            meta_gate_input_dim = fusion_input_size  # 256
            if self.use_gate:
                meta_gate_input_dim += 1  # + cosine bias
            self.meta_gate = nn.Linear(meta_gate_input_dim, 2)

            # Shared projector: 256 → text_in → pseudo_tokens
            self.shared_projector = nn.Linear(fusion_input_size, text_in)
            self.shared_token_projector = nn.Linear(1, args.pseudo_tokens)
        else:
            # ── Non-MoE Baseline (original multi_scale_fusion) ──
            self.mutli_scale_fusion = mutli_scale_fusion(
                input_size=fusion_input_size,
                output_size=text_in,
                pseudo_tokens=args.pseudo_tokens
            )

        # Optional: DiffLoss
        if self.use_diff_loss or self.use_expert_diff_loss:
            self.diff_loss_fn = DiffLoss()

        # Optional: NCE Loss
        if self.use_nce_loss:
            nce_hidden_dim = getattr(args, 'nce_hidden_dim', 32)
            nce_pred_steps = getattr(args, 'nce_pred_steps', 2)
            # text ↔ audio CPC
            self.cpc_text_audio = LightweightCrossCPC(
                text_dim=text_in,
                other_dim=args.a_lstm_hidden_size,
                nce_hidden_dim=nce_hidden_dim,
                n_prediction_steps=nce_pred_steps
            )
            # text ↔ video CPC
            self.cpc_text_video = LightweightCrossCPC(
                text_dim=text_in,
                other_dim=args.v_lstm_hidden_size,
                nce_hidden_dim=nce_hidden_dim,
                n_prediction_steps=nce_pred_steps
            )

    def _compute_lb_loss(self, gate_weights):
        """Entropy-based load-balance loss for N≥3 homogeneous experts."""
        expert_load = torch.mean(gate_weights, dim=0)  # [num_experts]
        lb_loss = -torch.sum(expert_load * torch.log(expert_load + 1e-10))
        return lb_loss

    def _compute_diff_loss_pairs(self, expert_outputs):
        """Compute pairwise DiffLoss between all expert outputs."""
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
        # 1. Global MoE
        global_out, global_gw, global_experts = self.global_moe(feature_f)  # [B, 256]

        # 2. Local MoE
        local_out, local_gw, local_experts = self.local_moe(feature_f)  # [B, 256]

        # 3. Meta-Gate
        meta_input = feature_f
        if self.use_gate:
            bias_score = F.cosine_similarity(audio_h, video_h, dim=-1).unsqueeze(-1)
            meta_input = torch.cat([feature_f, bias_score], dim=-1)
        meta_weights = F.softmax(self.meta_gate(meta_input), dim=-1)  # [B, 2]

        fused = meta_weights[:, 0].unsqueeze(1) * global_out + \
                meta_weights[:, 1].unsqueeze(1) * local_out  # [B, 256]

        # 4. Shared projector → [B, pseudo_tokens, text_in]
        projected = self.shared_projector(fused)       # [B, text_in]
        fusion_h = self.shared_token_projector(projected.unsqueeze(2))  # [B, text_in, pseudo_tokens]
        fusion_h = fusion_h.permute(0, 2, 1)           # [B, pseudo_tokens, text_in]

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

    def forward(self, labels, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text

        # Text embedding
        text_embed = self.LLM.text_embedding(text[:, 0, :].long())  # [B, L, text_in]

        # Audio & Video encoding
        if self.use_nce_loss:
            audio_h, audio_seq = self.audio_LSTM(audio, audio_len, return_sequence=True)
            video_h, video_seq = self.video_LSTM(video, video_len, return_sequence=True)
        else:
            audio_h = self.audio_LSTM(audio, audio_len)
            video_h = self.video_LSTM(video, video_len)

        # Shared mixer
        feature_f = self.text_guide_mixer(audio_h, video_h, text_embed)

        # Fusion
        if self.use_moe_fusion:
            fusion_h, moe_aux = self._dual_moe_forward(audio_h, video_h, feature_f)
        else:
            fusion_h = self.mutli_scale_fusion(feature_f)

        # LLM forward
        LLM_input = torch.cat([fusion_h, text_embed], dim=1)
        LLM_output = self.LLM(LLM_input, labels)

        res = {
            'Loss': LLM_output.loss,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': feature_f,
        }

        # ── Auxiliary Losses (only during training) ──
        if self.use_moe_fusion and self.training:
            # LB Loss (per branch)
            if self.use_moe_lb_loss:
                lb_global = self._compute_lb_loss(moe_aux['global_gw'])
                lb_local = self._compute_lb_loss(moe_aux['local_gw'])
                res['MoE_LB_Loss'] = (lb_global + lb_local) * 0.01

            # DiffLoss between branches
            if self.use_diff_loss:
                diff_branch = self.diff_loss_fn(moe_aux['global_out'], moe_aux['local_out'])
                res['DiffLoss'] = diff_branch * self.diff_loss_weight

            # DiffLoss within experts
            if self.use_expert_diff_loss:
                diff_global = self._compute_diff_loss_pairs(moe_aux['global_experts'])
                diff_local = self._compute_diff_loss_pairs(moe_aux['local_experts'])
                res['ExpertDiffLoss'] = (diff_global + diff_local) * self.diff_loss_weight

        # NCE Loss
        if self.use_nce_loss and self.training:
            nce_ta = self.cpc_text_audio(text_embed, audio_seq)
            nce_tv = self.cpc_text_video(text_embed, video_seq)
            res['NCELoss'] = (nce_ta + nce_tv) * self.nce_weight

        return res

    def generate(self, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text
        text_embed = self.LLM.text_embedding(text[:, 0, :].long())

        audio_h = self.audio_LSTM(audio, audio_len)
        video_h = self.video_LSTM(video, video_len)

        # Shared mixer
        feature_f = self.text_guide_mixer(audio_h, video_h, text_embed)

        if self.use_moe_fusion:
            fusion_h, _ = self._dual_moe_forward(audio_h, video_h, feature_f)
        else:
            fusion_h = self.mutli_scale_fusion(feature_f)

        LLM_input = torch.cat([fusion_h, text_embed], dim=1)
        LLM_output = self.LLM.generate(LLM_input)

        return LLM_output
