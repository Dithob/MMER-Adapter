# self supervised multimodal multi-task learning network
import math
import os
import sys
import collections
from torch.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.Textmodel import Language_model

__all__ = ['CMCM']

# ============================================================================
# Loss Modules
# ============================================================================

class DiffLoss(nn.Module):
    """Orthogonality loss: encourages two representations to be different (complementary).
    Ref: MMGAT_EMO/model_utils.py"""
    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        # Zero mean
        input1 = input1 - torch.mean(input1, dim=0, keepdim=True)
        input2 = input2 - torch.mean(input2, dim=0, keepdim=True)
        # L2 normalize
        input1_l2 = input1 / (torch.norm(input1, p=2, dim=1, keepdim=True) + 1e-6)
        input2_l2 = input2 / (torch.norm(input2, p=2, dim=1, keepdim=True) + 1e-6)
        # Inverse distance → smaller distance = higher loss
        diff_loss = 1.0 / (torch.mean(torch.norm(input1_l2 - input2_l2, p=2, dim=1)) + 1e-6)
        return diff_loss


class LightweightCrossCPC(nn.Module):
    """Lightweight Cross-modal CPC (Contrastive Predictive Coding).
    Contrasts text↔audio or text↔video using temporal features.
    Ref: MMGAT_EMO/CPC.py (Cross_CPC), adapted for MMER-Adapter."""
    def __init__(self, text_dim, other_dim, nce_hidden_dim=32, n_prediction_steps=2, min_start_steps=3):
        super().__init__()
        self.nce_hidden_dim = nce_hidden_dim
        self.n_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.lsoftmax = nn.LogSoftmax(dim=1)

        # Internal projections to unified nce_hidden_dim
        self.text_proj = nn.Linear(text_dim, nce_hidden_dim)
        self.other_proj = nn.Linear(other_dim, nce_hidden_dim)

        # Lightweight autoregressive LSTMs (1 layer)
        self.text_ar = nn.LSTM(nce_hidden_dim, nce_hidden_dim, 1, batch_first=True)
        self.other_ar = nn.LSTM(nce_hidden_dim, nce_hidden_dim, 1, batch_first=True)

        # Prediction heads
        self.text_predictors = nn.ModuleList([
            nn.Linear(nce_hidden_dim, nce_hidden_dim) for _ in range(n_prediction_steps)
        ])
        self.other_predictors = nn.ModuleList([
            nn.Linear(nce_hidden_dim, nce_hidden_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, text_seq, other_seq):
        """
        Args:
            text_seq:  [B, T_text, text_dim]   — LLM text embeddings
            other_seq: [B, T_other, other_dim] — LSTM temporal output (audio or video)
        Returns:
            nce_loss: scalar
        """
        # Project to unified space
        text_vq = self.text_proj(text_seq)     # [B, T_text, nce_hidden_dim]
        other_vq = self.other_proj(other_seq)  # [B, T_other, nce_hidden_dim]

        # Use the shorter sequence length for both
        T = min(text_vq.size(1), other_vq.size(1))
        text_vq = text_vq[:, :T, :]
        other_vq = other_vq[:, :T, :]

        batch_dim = text_vq.size(0)

        # Need at least min_start_steps + n_prediction_steps time steps
        if T < self.min_start_steps + self.n_steps + 1:
            return torch.tensor(0.0, device=text_vq.device, requires_grad=True)

        t_samples = (torch.randint(T - self.n_steps - self.min_start_steps, size=(1,)) + self.min_start_steps).long()

        nce = 0.0
        # Encode future samples
        text_encode = torch.stack([text_vq[:, t_samples + i, :].squeeze(1) for i in range(self.n_steps)])
        other_encode = torch.stack([other_vq[:, t_samples + i, :].squeeze(1) for i in range(self.n_steps)])

        # Autoregressive context
        text_forward = text_vq[:, :t_samples + 1, :]
        other_forward = other_vq[:, :t_samples + 1, :]

        h_text = (torch.zeros(1, batch_dim, self.nce_hidden_dim, device=text_vq.device),
                  torch.zeros(1, batch_dim, self.nce_hidden_dim, device=text_vq.device))
        h_other = (torch.zeros(1, batch_dim, self.nce_hidden_dim, device=other_vq.device),
                   torch.zeros(1, batch_dim, self.nce_hidden_dim, device=other_vq.device))

        text_context, _ = self.text_ar(text_forward, h_text)
        other_context, _ = self.other_ar(other_forward, h_other)

        text_context = text_context[:, -1, :]    # [B, nce_hidden_dim]
        other_context = other_context[:, -1, :]  # [B, nce_hidden_dim]

        # Predict and compute NCE
        for i in range(self.n_steps):
            text_pred = self.text_predictors[i](text_context)
            other_pred = self.other_predictors[i](other_context)

            # Cross-modal: text predicts other's future, other predicts text's future
            total1 = torch.mm(text_encode[i], other_pred.t())
            total2 = torch.mm(other_encode[i], text_pred.t())
            nce += torch.sum(torch.diag(self.lsoftmax(total1)))
            nce += torch.sum(torch.diag(self.lsoftmax(total2)))

        nce /= -1.0 * batch_dim * self.n_steps
        return 0.05 * nce


# ============================================================================
# MoE Modules
# ============================================================================

class GlobalMoE(nn.Module):
    """Global Emotion MoE: multi_scale_fusion upgraded to MoE.
    3 scale experts (coarse/fine/medium) with Gate replacing Integrating Conv."""
    def __init__(self, input_size=256, output_size=4096, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        multi_scale_hidden = 256

        # 3 scale experts (same structure as original multi_scale_fusion)
        self.experts = nn.ModuleList([
            nn.Sequential(  # Expert 1 (coarse): 256 → 512 → 256
                nn.Linear(input_size, output_size // 8),
                nn.GELU(),
                nn.Linear(output_size // 8, multi_scale_hidden)
            ),
            nn.Sequential(  # Expert 2 (fine): 256 → 128 → 256
                nn.Linear(input_size, output_size // 32),
                nn.GELU(),
                nn.Linear(output_size // 32, multi_scale_hidden)
            ),
            nn.Sequential(  # Expert 3 (medium): 256 → 256 → 256
                nn.Linear(input_size, output_size // 16),
                nn.GELU(),
                nn.Linear(output_size // 16, multi_scale_hidden)
            ),
        ])

        # If more experts requested, add extra bottleneck experts
        for _ in range(num_experts - 3):
            self.experts.append(nn.Sequential(
                nn.Linear(input_size, 64),
                nn.GELU(),
                nn.Linear(64, multi_scale_hidden)
            ))

        self.gate = nn.Linear(input_size, self.num_experts)

    def forward(self, x):
        """
        Args: x [B, 256]
        Returns: output [B, 256], gate_weights [B, num_experts], expert_outputs list
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        gate_weights = F.softmax(self.gate(x), dim=-1)  # [B, num_experts]
        expert_outputs = [expert(x) for expert in self.experts]  # list of [B, 256]

        # Weighted sum
        output = torch.zeros_like(expert_outputs[0])
        for i, e_out in enumerate(expert_outputs):
            output = output + gate_weights[:, i].unsqueeze(1) * e_out

        return output, gate_weights, expert_outputs


class LocalMoE(nn.Module):
    """Local Emotion MoE: lightweight bottleneck experts."""
    def __init__(self, input_size=256, num_experts=3, bottleneck_dim=64):
        super().__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, input_size)
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Linear(input_size, num_experts)

    def forward(self, x):
        """
        Args: x [B, 256]
        Returns: output [B, 256], gate_weights [B, num_experts], expert_outputs list
        """
        gate_weights = F.softmax(self.gate(x), dim=-1)
        expert_outputs = [expert(x) for expert in self.experts]

        output = torch.zeros_like(expert_outputs[0])
        for i, e_out in enumerate(expert_outputs):
            output = output + gate_weights[:, i].unsqueeze(1) * e_out

        return output, gate_weights, expert_outputs


# ============================================================================
# Sub-modules
# ============================================================================

class TVA_LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(TVA_LSTM, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 256)
        self.hidden_size = hidden_size

    def forward(self, x, lengths, return_sequence=False):
        packed_sequence = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        output, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze(0))
        h = self.linear(h)  # [B, 256]

        if return_sequence:
            output_padded, _ = pad_packed_sequence(output, batch_first=True)  # [B, T, hidden_size]
            return h, output_padded
        return h


class Text_guide_mixer(nn.Module):
    def __init__(self, text_in=4096):
        super(Text_guide_mixer, self).__init__()
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.text_mlp = nn.Linear(text_in, 256)
    def forward(self, audio, video, text):
        text_GAP = self.GAP(text.permute(0, 2, 1)).squeeze()
        text_knowledge = self.text_mlp(text_GAP)

        audio_mixed = torch.mul(audio, text_knowledge)
        video_mixed = torch.mul(video, text_knowledge)

        fusion = audio_mixed + video_mixed

        return fusion


class Lightweight_mixer(nn.Module):
    def __init__(self):
        super(Lightweight_mixer, self).__init__()
    def forward(self, audio, video, text):
        return audio + video


class mutli_scale_fusion(nn.Module):
    """Original multi-scale fusion (non-MoE baseline)."""
    def __init__(self, input_size, output_size, pseudo_tokens=4):
        super(mutli_scale_fusion, self).__init__()
        multi_scale_hidden = 256
        self.scale1 = nn.Sequential(
            nn.Linear(input_size, output_size // 8),
            nn.GELU(),
            nn.Linear(output_size // 8, multi_scale_hidden)
        )
        self.scale2 = nn.Sequential(
            nn.Linear(input_size, output_size // 32),
            nn.GELU(),
            nn.Linear(output_size // 32, multi_scale_hidden)
        )
        self.scale3 = nn.Sequential(
            nn.Linear(input_size, output_size // 16),
            nn.GELU(),
            nn.Linear(output_size // 16, multi_scale_hidden)
        )

        self.integrating = Integrating(scales=3)
        self.multi_scale_projector = nn.Linear(multi_scale_hidden, output_size)
        self.projector = nn.Linear(1, pseudo_tokens)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        scale1 = self.scale1(x)
        scale2 = self.scale2(x)
        scale3 = self.scale3(x)

        multi_scale_stack = torch.stack([scale1, scale2, scale3], dim=2)
        multi_scale_integrating = self.integrating(multi_scale_stack)

        multi_scale = self.multi_scale_projector(multi_scale_integrating)
        output = self.projector(multi_scale.unsqueeze(2))
        return output.permute(0, 2, 1)


class Integrating(nn.Module):
    def __init__(self, scales):
        super(Integrating, self).__init__()
        self.Integrating_layer = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(1, scales), stride=1))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.Integrating_layer(x)
        x = x.squeeze((1, 3))
        return x


# ============================================================================
# Main Model
# ============================================================================

class CMCM(nn.Module):
    def __init__(self, args):
        super(CMCM, self).__init__()
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
