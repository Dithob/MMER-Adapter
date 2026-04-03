import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AdaptiveModalMixer']


class AdaptiveModalMixer(nn.Module):
    """Adaptive Modal Mixer (AMM): symmetric multi-modal fusion.

    Treats audio_h, video_h, text_pooled as 3 equal "modal tokens",
    applies Self-Attention so any two modalities can directly interact,
    then uses adaptive weighted pooling to produce a fused output.

    Inspired by Emotion-LLaMA-v2's ConvProAttention, adapted for 1D vectors.
    """

    def __init__(self, text_in=4096, fusion_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim

        # Text projection (preserves TGM's GAP approach)
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.text_proj = nn.Linear(text_in, fusion_dim)

        # Self-Attention among 3 modal tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(fusion_dim)

        # FFN (post-attention refinement)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        self.norm2 = nn.LayerNorm(fusion_dim)

        # Adaptive modality importance weights
        self.modal_gate = nn.Linear(fusion_dim * 3, 3)

    def forward(self, audio_h, video_h, text_embed):
        """
        Args:
            audio_h:    [B, 256]
            video_h:    [B, 256]
            text_embed: [B, L, text_in]
        Returns:
            output:     [B, 256]
        """
        # 1. Text pooling + projection
        text_pooled = self.GAP(text_embed.permute(0, 2, 1)).squeeze(-1)  # [B, text_in]
        text_h = self.text_proj(text_pooled)  # [B, 256]

        # 2. Stack as 3 modal tokens → [B, 3, 256]
        modal_tokens = torch.stack([audio_h, video_h, text_h], dim=1)

        # 3. Self-Attention (audio↔video, audio↔text, video↔text)
        attn_out, _ = self.cross_attn(modal_tokens, modal_tokens, modal_tokens)
        modal_tokens = self.norm1(modal_tokens + attn_out)

        # 4. FFN
        modal_tokens = self.norm2(modal_tokens + self.ffn(modal_tokens))

        # 5. Adaptive weighted pooling
        concat_ctx = torch.cat([audio_h, video_h, text_h], dim=-1)  # [B, 768]
        weights = F.softmax(self.modal_gate(concat_ctx), dim=-1)    # [B, 3]
        output = (weights.unsqueeze(-1) * modal_tokens).sum(dim=1)  # [B, 256]

        return output
