import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AdaptiveModalMixer']


class AdaptiveModalMixer(nn.Module):
    """Adaptive Modal Mixer (AMM): symmetric multi-modal fusion with TCAP.

    Treats audio_h, video_h, text_pooled as 3 equal "modal tokens",
    applies Self-Attention so any two modalities can directly interact,
    then uses adaptive weighted pooling to produce a fused output.

    TCAP (Text Confidence-Aware Attention Prior) injects a sample-adaptive
    text bias into the attention, so that when text emotion is clear the
    model focuses more on text, while when text is ambiguous A/V features
    can contribute more.

    Inspired by Emotion-LLaMA-v2's ConvProAttention, adapted for 1D vectors.
    """

    def __init__(self, text_in=4096, fusion_dim=256, num_heads=4, dropout=0.1,
                 use_tcap=True):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.use_tcap = use_tcap

        # Text projection (preserves TGM's GAP approach)
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.text_proj = nn.Linear(text_in, fusion_dim)

        # ── TCAP: Text Confidence-Aware Attention Prior ──
        if self.use_tcap:
            self.text_confidence = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 4),
                nn.GELU(),
                nn.Linear(fusion_dim // 4, 1),
            )

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
            aux:        dict with 'align_loss' and 'text_confidence'
        """
        # 1. Text pooling + projection
        text_pooled = self.GAP(text_embed.permute(0, 2, 1)).squeeze(-1)  # [B, text_in]
        text_h = self.text_proj(text_pooled)  # [B, 256]

        # 2. Stack as 3 modal tokens → [B, 3, 256]
        #    Order: [audio, video, text] — text is at index 2
        modal_tokens = torch.stack([audio_h, video_h, text_h], dim=1)

        # 3. TCAP: compute text confidence and build attention bias
        attn_mask = None
        text_conf = None
        if self.use_tcap:
            text_conf = torch.sigmoid(self.text_confidence(text_h))  # [B, 1]
            # Build additive attention bias: all tokens attend more to text (column 2)
            # Shape [B*num_heads, 3, 3] for MultiheadAttention
            B = audio_h.shape[0]
            num_heads = self.cross_attn.num_heads
            attn_mask = torch.zeros(B, 3, 3, device=audio_h.device, dtype=audio_h.dtype)
            # Bias the text column (index 2) — all queries attend more to text
            attn_mask[:, :, 2] = text_conf.squeeze(-1).unsqueeze(1).expand(-1, 3)
            # Expand for multi-head: [B, 3, 3] → [B*num_heads, 3, 3]
            attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
            attn_mask = attn_mask.reshape(B * num_heads, 3, 3)

        # 4. Self-Attention (audio↔video, audio↔text, video↔text) with TCAP bias
        attn_out, _ = self.cross_attn(modal_tokens, modal_tokens, modal_tokens,
                                       attn_mask=attn_mask)
        modal_tokens = self.norm1(modal_tokens + attn_out)

        # 5. FFN
        modal_tokens = self.norm2(modal_tokens + self.ffn(modal_tokens))

        # 6. Adaptive weighted pooling
        concat_ctx = torch.cat([audio_h, video_h, text_h], dim=-1)  # [B, 768]
        weights = F.softmax(self.modal_gate(concat_ctx), dim=-1)    # [B, 3]
        output = (weights.unsqueeze(-1) * modal_tokens).sum(dim=1)  # [B, 256]

        # 7. Compute auxiliary losses
        # align_loss: encourage modal tokens to be aligned after attention
        a_out, v_out, t_out = modal_tokens[:, 0], modal_tokens[:, 1], modal_tokens[:, 2]
        align_loss = 0.5 * (
            (t_out - a_out).pow(2).mean() +
            (t_out - v_out).pow(2).mean() +
            (a_out - v_out).pow(2).mean()
        )

        aux = {
            'align_loss': align_loss,
            'text_confidence': text_conf,
        }

        return output, aux
