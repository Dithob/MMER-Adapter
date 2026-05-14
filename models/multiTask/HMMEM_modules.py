import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = [
    'TVA_LSTM',
    'Text_guide_mixer',
    'Lightweight_mixer',
    'mutli_scale_fusion',
    'Integrating',
    'FeatureAdapter',
    'ATGFBFF',
    'MultiScaleLatentAttentionFusion',
    'SharedOffsetFusion',
]


class FeatureAdapter(nn.Module):
    """Lightweight adapter to reduce high-dim encoder features (HuBERT/Whisper)
    to a dimension suitable for LSTM processing.
    Only instantiated when input feature dim > adapter_dim.
    """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x: [B, T, in_dim] → [B, T, out_dim]
        return self.adapter(x)

class TVA_LSTM(nn.Module):
    """Temporal encoder for audio/video features.

    Supports two modes controlled by `use_bilstm`:
      - sLSTM (default): unidirectional LSTM, pools via final hidden state.
      - BiLSTM: bidirectional LSTM + learned attention pooling over all
        timesteps, preserving richer temporal information.
    """
    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2,
                 bidirectional=False, use_bilstm=False):
        super(TVA_LSTM, self).__init__()
        self.use_bilstm = use_bilstm or bidirectional
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(
            in_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.use_bilstm,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        if self.use_bilstm:
            # BiLSTM output dim = hidden_size * 2 (forward + backward)
            rnn_out_dim = hidden_size * 2
            # Learned attention pooling: query vector + projection
            self.attn_query = nn.Parameter(torch.randn(rnn_out_dim) * 0.02)
            self.attn_proj = nn.Linear(rnn_out_dim, rnn_out_dim)
            self.linear = nn.Linear(rnn_out_dim, 256)
        else:
            self.linear = nn.Linear(hidden_size, 256)

    def _attention_pool(self, output_padded, lengths):
        """Learned attention pooling over BiLSTM sequence outputs.

        Args:
            output_padded: [B, T_max, rnn_out_dim]
            lengths:       [B] actual sequence lengths
        Returns:
            pooled: [B, rnn_out_dim]
        """
        # Project and compute attention scores: [B, T_max]
        projected = torch.tanh(self.attn_proj(output_padded))  # [B, T, D]
        scores = torch.matmul(projected, self.attn_query)       # [B, T]

        # Mask padding positions with -inf before softmax
        max_len = output_padded.size(1)
        mask = torch.arange(max_len, device=output_padded.device).unsqueeze(0)  # [1, T]
        mask = mask >= lengths.unsqueeze(1)  # [B, T], True = pad
        scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        pooled = (attn_weights * output_padded).sum(dim=1)          # [B, D]
        return pooled

    def forward(self, x, lengths, return_sequence=False):
        packed_sequence = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        output, final_states = self.rnn(packed_sequence)
        output_padded, _ = pad_packed_sequence(output, batch_first=True)  # [B, T, D]

        if self.use_bilstm:
            # Attention pooling over full BiLSTM output
            h = self._attention_pool(output_padded, lengths.to(output_padded.device))
            h = self.dropout(h)
        else:
            # Original sLSTM path: use final hidden state
            h = self.dropout(final_states[0].squeeze(0))

        h = self.linear(h)  # [B, 256]

        if return_sequence:
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


class ATGFBFF(nn.Module):
    """Adaptive Text-Guided Fiber Bundle Feature Fusion (论文 §3.3).

    Projects text/audio/video into shared and private spaces, then forms a
    shared semantic core plus modality-specific fiber offsets.

    When use_mslaf=True, the fused representation M is further refined through
    MultiScaleLatentAttentionFusion (论文 §3.4) before pseudo-token generation.

    Output is a pseudo-token sequence shaped [B, pseudo_tokens, hidden_size]
    so it can be directly consumed by the frozen LLM prompt wrapper.
    """

    def __init__(self, input_size=256, hidden_size=256, pseudo_tokens=4,
                 dropout=0.1, use_mslaf=False, num_latents=4, latent_size=None):
        super().__init__()
        self.pseudo_tokens = pseudo_tokens
        self.use_mslaf = use_mslaf

        # ── 共享 / 特有投影层 (论文 §3.3: z_c = V̄·W) ──
        # 简化为 Linear + LayerNorm，匹配论文的线性投影设计
        self.text_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.audio_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.audio_private = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.video_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.video_private = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # ── 可学习模态权重 (论文: α_logits → softmax → (λ_T, λ_V, λ_A)) ──
        # 初始化偏向文本 (论文: "initializing by maximizing λ_T")
        self.modality_logits = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))

        # ── 伪词元生成 ──
        if use_mslaf:
            # 论文完整流程: M → MSLAF → L' → Z_M → offset pseudo-tokens
            _latent_size = latent_size or hidden_size
            self.mslaf = MultiScaleLatentAttentionFusion(
                input_size=hidden_size,
                latent_size=_latent_size,
                num_latents=num_latents,
                dropout=dropout,
            )
            # 论文 §3.5: Z_M = Linear(L'), T_M[b,j,:] = Z_M[b,:] + E[j,:]
            self.mslaf_core_proj = nn.Linear(_latent_size, hidden_size)
            self.mslaf_token_offset = nn.Parameter(
                torch.randn(pseudo_tokens, hidden_size) * 0.02
            )
        else:
            # 简化路径: M → core_proj → offset pseudo-tokens
            self.core_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.token_offset = nn.Parameter(
                torch.randn(pseudo_tokens, hidden_size) * 0.02
            )

    def forward(self, audio, video, text):
        """
        Args:
            audio: [B, H] — sLSTM audio encoding
            video: [B, H] — sLSTM video encoding
            text:  [B, H] — pooled text embedding
        Returns:
            fusion_h: [B, pseudo_tokens, hidden_size]
            aux: dict with align_loss, fiber_loss, shared/fiber features, weights
        """
        # ── 共享 / 特有投影 ──
        z_c_t = self.text_shared(text)
        z_c_a = self.audio_shared(audio)
        z_p_a = self.audio_private(audio)
        z_c_v = self.video_shared(video)
        z_p_v = self.video_private(video)

        # ── 自适应加权共享语义核 Z_s ──
        weights = torch.softmax(self.modality_logits, dim=0)
        z_s = weights[0] * z_c_t + weights[1] * z_c_a + weights[2] * z_c_v

        # ── 纤维偏移 ──
        delta_a = z_p_a - z_s
        delta_v = z_p_v - z_s

        # ── 融合表示 M = Z_s + ΔV + ΔA ──
        fused = z_s + delta_a + delta_v

        # ── 伪词元生成 ──
        if self.use_mslaf:
            # 论文完整流程: M → MSLAF → L' → Z_M + E[j]
            L_prime = self.mslaf(fused)                         # [B, n_l, d_l]
            Z_M = self.mslaf_core_proj(L_prime.mean(dim=1))     # [B, H]
            fusion_h = Z_M.unsqueeze(1) + self.mslaf_token_offset.unsqueeze(0)
        else:
            # 简化路径: M → core_proj → Z_M + E[j]
            core = self.core_proj(fused)
            fusion_h = core.unsqueeze(1) + self.token_offset.unsqueeze(0)

        # ── 辅助损失 ──
        aux = {
            'align_loss': 0.5 * (
                (z_c_t - z_c_a).pow(2).mean() +
                (z_c_t - z_c_v).pow(2).mean() +
                (z_c_a - z_c_v).pow(2).mean()
            ),
            'fiber_loss': delta_a.pow(2).mean() + delta_v.pow(2).mean(),
            'shared_text': z_c_t,
            'shared_audio': z_c_a,
            'shared_video': z_c_v,
            'fiber_audio': delta_a,
            'fiber_video': delta_v,
            'weights': weights,
        }
        return fusion_h, aux


class MultiScaleLatentAttentionFusion(nn.Module):
    """Multi-Scale Latent Attention Fusion (论文 §3.4).

    M → 三路 bottleneck MLP(r∈{8,16,32}) → latent projection(d_l)
    → LN → concat as KV → learnable latents as Q → cross-attention
    → 残差 + FFN → L'

    Args:
        input_size:  融合表示 M 的维度 H
        latent_size: 潜在空间维度 d_l（默认与 input_size 相同）
        num_latents: 可学习潜变量数量 n_l
        bottleneck_factors: 三路 MLP 的压缩因子 r（论文取 {8, 16, 32}）
        num_heads:   cross-attention 头数
        dropout:     dropout 率
    """

    def __init__(self, input_size=256, latent_size=256, num_latents=4,
                 bottleneck_factors=(8, 16, 32), num_heads=4, dropout=0.1):
        super().__init__()
        self.num_latents = num_latents

        # 三路 bottleneck MLP: M → W1(H→H/r) → GELU → W2(H/r→H)
        self.scale_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, input_size // r),
                nn.GELU(),
                nn.Linear(input_size // r, input_size),
            ) for r in bottleneck_factors
        ])

        # 低秩潜在投影: H → d_l
        self.latent_projs = nn.ModuleList([
            nn.Linear(input_size, latent_size) for _ in bottleneck_factors
        ])

        # Latent Normalization (LN) — per-branch
        self.latent_norms = nn.ModuleList([
            nn.LayerNorm(latent_size) for _ in bottleneck_factors
        ])

        # 可学习潜变量 L ∈ R^{n_l × d_l}
        self.latents = nn.Parameter(torch.randn(num_latents, latent_size) * 0.02)

        # Cross-attention: Q = L, KV = concat(LN(m'_1), LN(m'_2), LN(m'_3))
        self.cross_attn = nn.MultiheadAttention(
            latent_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        # 残差 + FFN (论文: N(L + M̄) + ψ(N(L + M̄)))
        self.norm1 = nn.LayerNorm(latent_size)
        self.ffn = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * 4, latent_size),
        )
        self.norm2 = nn.LayerNorm(latent_size)

    def forward(self, fused):
        """
        Args:
            fused: [B, H] — ATGFBFF 融合输出 M
        Returns:
            L_prime: [B, num_latents, latent_size] — 精炼后的潜在状态
        """
        # 三路多尺度 MLP + 低秩投影 + LN
        kv_parts = []
        for mlp, proj, norm in zip(self.scale_mlps, self.latent_projs, self.latent_norms):
            m_i = mlp(fused)                        # [B, H]
            m_i_proj = norm(proj(m_i))              # [B, d_l] → LN
            kv_parts.append(m_i_proj.unsqueeze(1))  # [B, 1, d_l]

        kv = torch.cat(kv_parts, dim=1)  # [B, 3, d_l]

        # Cross-attention: learnable latents as Q
        Q = self.latents.unsqueeze(0).expand(fused.size(0), -1, -1)  # [B, n_l, d_l]
        attn_out, _ = self.cross_attn(Q, kv, kv, need_weights=False)

        # 残差 + LayerNorm + FFN + 残差 + LayerNorm
        L_prime = self.norm1(Q + attn_out)
        L_prime = self.norm2(L_prime + self.ffn(L_prime))  # [B, n_l, d_l]

        return L_prime


class SharedOffsetFusion(nn.Module):
    """Shared semantic core + micro offset fusion.

    Designed as a lightweight, switchable bridge between ATGFB-style decomposition
    and the existing MMER dual-path setup.

    When use_mslaf=True, the fused output is refined through MSLAF before
    pseudo-token generation.
    """

    def __init__(self, input_size=256, hidden_size=256, pseudo_tokens=4,
                 dropout=0.1, mode='gate', use_mslaf=False, num_latents=4,
                 latent_size=None):
        super().__init__()
        self.mode = mode
        self.use_mslaf = use_mslaf

        # ── 共享 / 特有投影层 (简化: Linear + LN) ──
        self.text_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.audio_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.video_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.audio_private = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.video_private = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # 初始化偏向文本
        self.modality_logits = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        # ── 伪词元生成 ──
        if use_mslaf:
            _latent_size = latent_size or hidden_size
            self.mslaf = MultiScaleLatentAttentionFusion(
                input_size=hidden_size,
                latent_size=_latent_size,
                num_latents=num_latents,
                dropout=dropout,
            )
            self.mslaf_core_proj = nn.Linear(_latent_size, hidden_size)
            self.mslaf_token_offset = nn.Parameter(
                torch.randn(pseudo_tokens, hidden_size) * 0.02
            )
        else:
            self.core_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.token_offset = nn.Parameter(
                torch.randn(pseudo_tokens, hidden_size) * 0.02
            )

    def forward(self, audio, video, text):
        if text.dim() == 3:
            text = text.mean(dim=1)

        z_c_t = self.text_shared(text)
        z_c_a = self.audio_shared(audio)
        z_c_v = self.video_shared(video)
        z_p_a = self.audio_private(audio)
        z_p_v = self.video_private(video)

        weights = torch.softmax(self.modality_logits, dim=0)
        z_shared = weights[0] * z_c_t + weights[1] * z_c_a + weights[2] * z_c_v
        delta_a = z_p_a - z_shared
        delta_v = z_p_v - z_shared
        delta_micro = delta_a + delta_v

        if self.mode == 'add':
            fused_core = z_shared + delta_micro
        elif self.mode == 'residual':
            fused_core = z_shared + 0.5 * delta_micro
        else:
            gate = torch.sigmoid(self.fusion_gate(torch.cat([z_shared, delta_micro], dim=-1)))
            fused_core = gate * z_shared + (1 - gate) * delta_micro

        # ── 伪词元生成 ──
        if self.use_mslaf:
            L_prime = self.mslaf(fused_core)
            Z_M = self.mslaf_core_proj(L_prime.mean(dim=1))
            fusion_h = Z_M.unsqueeze(1) + self.mslaf_token_offset.unsqueeze(0)
        else:
            core = self.core_proj(fused_core)
            fusion_h = core.unsqueeze(1) + self.token_offset.unsqueeze(0)

        align_loss = 0.5 * (
            (z_c_t - z_c_a).pow(2).mean() +
            (z_c_t - z_c_v).pow(2).mean() +
            (z_c_a - z_c_v).pow(2).mean()
        )
        offset_loss = delta_a.pow(2).mean() + delta_v.pow(2).mean()

        aux = {
            'align_loss': align_loss,
            'offset_loss': offset_loss,
            'shared_text': z_c_t,
            'shared_audio': z_c_a,
            'shared_video': z_c_v,
            'delta_audio': delta_a,
            'delta_video': delta_v,
            'weights': weights,
            'fused_core': fused_core,
        }
        return fusion_h, aux
