import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['OriginAMM', 'AdaptiveModalMixer']


class OriginAMM(nn.Module):
    """Original Adaptive Modal Mixer with TCAP (preserved for ablation).

    Treats audio_h, video_h, text_pooled as 3 equal "modal tokens",
    applies Self-Attention (with optional TCAP bias) so any two modalities
    can directly interact, then uses adaptive weighted pooling.
    """

    def __init__(self, text_in=4096, fusion_dim=256, num_heads=4, dropout=0.1,
                 use_tcap=True):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.use_tcap = use_tcap

        # Text projection
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.text_proj = nn.Linear(text_in, fusion_dim)

        # TCAP: Text Confidence-Aware Attention Prior
        if self.use_tcap:
            self.text_confidence = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 4),
                nn.GELU(),
                nn.Linear(fusion_dim // 4, 1),
            )

        # Self-Attention among 3 modal tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(fusion_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(fusion_dim * 2, fusion_dim)
        )
        self.norm2 = nn.LayerNorm(fusion_dim)

        # Adaptive modality importance weights
        self.modal_gate = nn.Linear(fusion_dim * 3, 3)

    def forward(self, audio_h, video_h, text_embed):
        text_pooled = self.GAP(text_embed.permute(0, 2, 1)).squeeze(-1)
        text_h = self.text_proj(text_pooled)

        modal_tokens = torch.stack([audio_h, video_h, text_h], dim=1)

        # TCAP bias
        attn_mask = None
        text_conf = None
        if self.use_tcap:
            text_conf = torch.sigmoid(self.text_confidence(text_h))
            B = audio_h.shape[0]
            num_heads = self.cross_attn.num_heads
            attn_mask = torch.zeros(B, 3, 3, device=audio_h.device, dtype=audio_h.dtype)
            attn_mask[:, :, 2] = text_conf.squeeze(-1).unsqueeze(1).expand(-1, 3)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
            attn_mask = attn_mask.reshape(B * num_heads, 3, 3)

        attn_out, _ = self.cross_attn(modal_tokens, modal_tokens, modal_tokens,
                                       attn_mask=attn_mask)
        modal_tokens = self.norm1(modal_tokens + attn_out)
        modal_tokens = self.norm2(modal_tokens + self.ffn(modal_tokens))

        concat_ctx = torch.cat([audio_h, video_h, text_h], dim=-1)
        weights = F.softmax(self.modal_gate(concat_ctx), dim=-1)
        output = (weights.unsqueeze(-1) * modal_tokens).sum(dim=1)

        # Align loss
        a_out, v_out, t_out = modal_tokens[:, 0], modal_tokens[:, 1], modal_tokens[:, 2]
        align_loss = 0.5 * (
            (t_out - a_out).pow(2).mean() +
            (t_out - v_out).pow(2).mean() +
            (a_out - v_out).pow(2).mean()
        )

        aux = {'align_loss': align_loss, 'text_confidence': text_conf}
        return output, aux


class AdaptiveModalMixer(nn.Module):
    """Improved AMM with switchable interaction modes.

    Supports three modes via ``amm_mode``:
    - ``base``:        Same as OriginAMM behaviour (flat self-attention + optional TCAP).
    - ``hierarchical``: Stage-1 text-anchored cross-attention + Stage-2 self-attention (H-AMM).
    - ``prototype``:   Learnable emotion prototype tokens participate in self-attention (EP-AMM).
    """

    def __init__(self, text_in=4096, fusion_dim=256, num_heads=4, dropout=0.1,
                 use_tcap=True, amm_mode='base', num_emotion_prototypes=4):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.use_tcap = use_tcap
        self.amm_mode = amm_mode

        # ── Text projection ──
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.text_proj = nn.Linear(text_in, fusion_dim)

        # ── TCAP ──
        if self.use_tcap:
            self.text_confidence = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 4), nn.GELU(),
                nn.Linear(fusion_dim // 4, 1),
            )

        # ── Stage-1: Text-Anchored Cross-Attention (hierarchical mode) ──
        if amm_mode == 'hierarchical':
            self.text_anchor_attn = nn.MultiheadAttention(
                embed_dim=fusion_dim, num_heads=num_heads,
                dropout=dropout, batch_first=True
            )
            self.norm0 = nn.LayerNorm(fusion_dim)

        # ── Emotion prototypes (prototype mode) ──
        if amm_mode == 'prototype':
            self.emotion_prototypes = nn.Parameter(
                torch.randn(num_emotion_prototypes, fusion_dim) * 0.02
            )

        # ── Stage-2: Tri-modal Self-Attention ──
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(fusion_dim)

        # ── FFN ──
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(fusion_dim * 2, fusion_dim)
        )
        self.norm2 = nn.LayerNorm(fusion_dim)

        # ── Adaptive modality gate ──
        self.modal_gate = nn.Linear(fusion_dim * 3, 3)

    def forward(self, audio_h, video_h, text_embed):
        # Text pooling + projection
        text_pooled = self.GAP(text_embed.permute(0, 2, 1)).squeeze(-1)
        text_h = self.text_proj(text_pooled)

        # ── Hierarchical: Text-Anchored alignment first ──
        if self.amm_mode == 'hierarchical':
            text_q = text_h.unsqueeze(1)                          # [B, 1, D]
            av_kv = torch.stack([audio_h, video_h], dim=1)        # [B, 2, D]
            text_aligned, _ = self.text_anchor_attn(text_q, av_kv, av_kv)
            text_h = self.norm0(text_q + text_aligned).squeeze(1) # [B, D]

        # Stack 3 modal tokens
        modal_tokens = torch.stack([audio_h, video_h, text_h], dim=1)

        # ── TCAP bias ──
        attn_mask = None
        text_conf = None
        if self.use_tcap:
            text_conf = torch.sigmoid(self.text_confidence(text_h))
            B = audio_h.shape[0]
            num_heads = self.cross_attn.num_heads
            attn_mask = torch.zeros(B, 3, 3, device=audio_h.device, dtype=audio_h.dtype)
            attn_mask[:, :, 2] = text_conf.squeeze(-1).unsqueeze(1).expand(-1, 3)
            # For prototype mode, TCAP bias only applies to the 3 modal tokens
            # (prototypes are appended after, so the mask is extended below)

        # ── Prototype mode: add emotion prototypes ──
        if self.amm_mode == 'prototype':
            B = modal_tokens.size(0)
            K = self.emotion_prototypes.size(0)
            proto = self.emotion_prototypes.unsqueeze(0).expand(B, -1, -1)
            tokens_with_proto = torch.cat([modal_tokens, proto], dim=1)  # [B, 3+K, D]

            # Extend TCAP mask for prototype tokens (no TCAP bias on prototypes)
            if attn_mask is not None:
                num_heads = self.cross_attn.num_heads
                full_mask = torch.zeros(B, 3 + K, 3 + K, device=audio_h.device, dtype=audio_h.dtype)
                full_mask[:, :3, :3] = attn_mask[:B]  # reuse the 3x3 part (before head expansion)
                attn_mask = full_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
                attn_mask = attn_mask.reshape(B * num_heads, 3 + K, 3 + K)
            
            attn_out, _ = self.cross_attn(tokens_with_proto, tokens_with_proto,
                                           tokens_with_proto, attn_mask=attn_mask)
            attn_out = attn_out[:, :3, :]  # only keep modal tokens
        else:
            # Expand TCAP mask for multi-head
            if attn_mask is not None:
                B = audio_h.shape[0]
                num_heads = self.cross_attn.num_heads
                attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
                attn_mask = attn_mask.reshape(B * num_heads, 3, 3)

            attn_out, _ = self.cross_attn(modal_tokens, modal_tokens, modal_tokens,
                                           attn_mask=attn_mask)

        modal_tokens = self.norm1(modal_tokens + attn_out)
        modal_tokens = self.norm2(modal_tokens + self.ffn(modal_tokens))

        # Adaptive weighted pooling
        concat_ctx = torch.cat([audio_h, video_h, text_h], dim=-1)
        weights = F.softmax(self.modal_gate(concat_ctx), dim=-1)
        output = (weights.unsqueeze(-1) * modal_tokens).sum(dim=1)

        # Align loss
        a_out, v_out, t_out = modal_tokens[:, 0], modal_tokens[:, 1], modal_tokens[:, 2]
        align_loss = 0.5 * (
            (t_out - a_out).pow(2).mean() +
            (t_out - v_out).pow(2).mean() +
            (a_out - v_out).pow(2).mean()
        )

        aux = {'align_loss': align_loss, 'text_confidence': text_conf}
        return output, aux
