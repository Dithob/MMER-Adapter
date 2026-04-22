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


class ATGFBFF(nn.Module):
    """Adaptive Text-Guided Fiber Bundle Feature Fusion.

    Projects text/audio/video into shared and private spaces, then forms a
    shared semantic core plus modality-specific fiber offsets.

    Output is a pseudo-token sequence shaped [B, pseudo_tokens, hidden_size]
    so it can be directly consumed by the frozen LLM prompt wrapper.
    """

    def __init__(self, input_size=256, hidden_size=256, pseudo_tokens=4, dropout=0.1):
        super().__init__()
        self.pseudo_tokens = pseudo_tokens
        self.text_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.audio_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.audio_private = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.video_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.video_private = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.modality_logits = nn.Parameter(torch.zeros(3))
        self.core_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_offset = nn.Parameter(torch.zeros(pseudo_tokens, hidden_size))

    def forward(self, audio, video, text):
        z_c_t = self.text_shared(text)
        z_c_a = self.audio_shared(audio)
        z_p_a = self.audio_private(audio)
        z_c_v = self.video_shared(video)
        z_p_v = self.video_private(video)

        weights = torch.softmax(self.modality_logits, dim=0)
        z_s = weights[0] * z_c_t + weights[1] * z_c_a + weights[2] * z_c_v
        delta_a = z_p_a - z_s
        delta_v = z_p_v - z_s
        fused = z_s + delta_a + delta_v
        core = self.core_proj(fused)
        fusion_h = core.unsqueeze(1) + self.token_offset.unsqueeze(0)

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
    """Multi-Scale Latent Attention Fusion.

    Uses multi-branch MLPs, latent projection, and cross-attention over learnable
    latent queries to build compact multimodal tokens.
    """

    def __init__(self, input_size=256, latent_size=256, num_latents=4, dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, input_size // 8),
                nn.GELU(),
                nn.Linear(input_size // 8, latent_size),
            ),
            nn.Sequential(
                nn.Linear(input_size, input_size // 16),
                nn.GELU(),
                nn.Linear(input_size // 16, latent_size),
            ),
            nn.Sequential(
                nn.Linear(input_size, input_size // 32),
                nn.GELU(),
                nn.Linear(input_size // 32, latent_size),
            ),
        ])
        self.latent_proj = nn.ModuleList([nn.Linear(latent_size, latent_size) for _ in range(3)])
        self.latents = nn.Parameter(torch.randn(num_latents, latent_size) * 0.02)
        self.attn = nn.MultiheadAttention(latent_size, num_heads=4, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(latent_size)
        self.ffn = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * 2, latent_size),
        )
        self.norm2 = nn.LayerNorm(latent_size)
        self.out_proj = nn.Linear(latent_size, input_size)
        self.pseudo_tokens = 4
        self.token_offset = nn.Parameter(torch.zeros(self.pseudo_tokens, input_size))

    def forward(self, fused):
        scale_feats = []
        for branch, proj in zip(self.branches, self.latent_proj):
            scale_feats.append(proj(branch(fused)))

        kv = torch.cat(scale_feats, dim=1)
        q = self.latents.unsqueeze(0).expand(fused.size(0), -1, -1)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        latent_state = self.norm1(q + attn_out)
        latent_state = self.norm2(latent_state + self.ffn(latent_state))

        core = self.out_proj(latent_state.mean(dim=1))
        pseudo_tokens = core.unsqueeze(1) + self.token_offset.unsqueeze(0)
        return pseudo_tokens


class SharedOffsetFusion(nn.Module):
    """Shared semantic core + micro offset fusion.

    Designed as a lightweight, switchable bridge between ATGFB-style decomposition
    and the existing MMER dual-path setup.
    """

    def __init__(self, input_size=256, hidden_size=256, pseudo_tokens=4, dropout=0.1, mode='gate'):
        super().__init__()
        self.mode = mode
        self.text_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.audio_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.video_shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.audio_private = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.video_private = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.modality_logits = nn.Parameter(torch.zeros(3))
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.core_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_offset = nn.Parameter(torch.zeros(pseudo_tokens, hidden_size))

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
