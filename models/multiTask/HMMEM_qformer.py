"""QFormer & CrossAttnExpander — multimodal-to-LLM pseudo-token generators.

Two modules live in this file:

1. **QFormerBridge** (满血 QFormer)
   - Receives *full temporal sequences* from Audio/Video LSTM as KV.
   - Learnable query tokens attend to the complete AV sequence via cross-attention.
   - Replaces **both** Mixer and Fusion in the pipeline.
   - Enabled by ``--use_qformer``.

2. **CrossAttnExpander** (legacy, lightweight ablation baseline)
   - Receives a *single fused vector* ``[B, 256]`` from Mixer output.
   - Expands it to pseudo-tokens via multi-scale projection + small cross-attention.
   - Sits in the **Fusion** layer (parallel to MSF / SD-MoE).
   - Enabled by ``--use_cross_attn_expander``.

Both share the same ``QFormerLayer`` building block.
"""

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════
# Shared building block
# ══════════════════════════════════════════════════════════════

class QFormerLayer(nn.Module):
    """Single QFormer layer: Self-Attention + Cross-Attention + FFN with residual + LayerNorm."""

    def __init__(self, d_model=256, num_heads=4, dropout=0.1):
        super().__init__()

        # Self-attention among query tokens (allows inter-token communication)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: queries attend to KV (projected input features)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, kv, kv_mask=None):
        """
        Args:
            queries: [B, num_queries, d_model]
            kv:      [B, kv_len, d_model]
            kv_mask: [B, kv_len] bool mask (True = **ignore** this position)
                     Compatible with nn.MultiheadAttention key_padding_mask.
        Returns:
            queries: [B, num_queries, d_model] (refined)
        """
        # Self-attention among queries
        q_sa, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.norm1(queries + q_sa)

        # Cross-attention: queries attend to input features
        q_ca, _ = self.cross_attn(
            queries, kv, kv, key_padding_mask=kv_mask, need_weights=False
        )
        queries = self.norm2(queries + q_ca)

        # FFN
        queries = self.norm3(queries + self.ffn(queries))

        return queries


# ══════════════════════════════════════════════════════════════
# 1. QFormerBridge — 满血 QFormer (replaces Mixer + Fusion)
# ══════════════════════════════════════════════════════════════

class QFormerBridge(nn.Module):
    """Full QFormer-style modality bridge operating on temporal sequences.

    Receives the **complete** audio and video LSTM output sequences as KV,
    and uses learnable query tokens to selectively extract information via
    cross-attention, producing pseudo-tokens aligned with the LLM embedding space.

    This module **replaces both Mixer and Fusion** in the pipeline when enabled.

    Architecture:
        1. Separate input projections for audio & video sequences → d_model
        2. Concatenate as unified KV: [B, T_a + T_v, d_model]
        3. Learnable query tokens: [1, num_queries, d_model]
        4. N layers of: Self-Attn(queries) + Cross-Attn(Q=queries, KV=av_seq) + FFN
        5. Output projection: [B, num_queries, d_model] → [B, num_queries, output_dim]

    Args:
        audio_dim:    dimension of audio LSTM output (default: 256)
        video_dim:    dimension of video LSTM output (default: 256)
        output_dim:   dimension of output pseudo-tokens = LLM hidden size
        num_queries:  number of learnable query tokens (default: 8)
        d_model:      internal transformer dimension (default: 256)
        num_layers:   number of cross-attention layers (default: 4)
        num_heads:    number of attention heads (default: 8)
        dropout:      dropout rate (default: 0.1)
    """

    def __init__(self, audio_dim=256, video_dim=256, output_dim=4096,
                 num_queries=8, d_model=256, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        # ── Input projections: map each modality to d_model ──
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Modality type embeddings (learned, added to KV) ──
        self.audio_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.video_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ── Learnable query tokens ──
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, d_model) * 0.02
        )

        # ── Cross-Attention + FFN layers ──
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(QFormerLayer(d_model, num_heads, dropout))

        # ── Output projection to LLM hidden dimension ──
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, audio_seq=None, video_seq=None,
                audio_mask=None, video_mask=None):
        """
        Args:
            audio_seq:  [B, T_a, audio_dim] — full audio LSTM output sequence (or None)
            video_seq:  [B, T_v, video_dim] — full video LSTM output sequence (or None)
            audio_mask: [B, T_a] — True where audio is padding (or None)
            video_mask: [B, T_v] — True where video is padding (or None)

        Returns:
            pseudo_tokens: [B, num_queries, output_dim] — for LLM input
        """
        kv_parts = []
        mask_parts = []

        if audio_seq is not None:
            B = audio_seq.shape[0]
            a_proj = self.audio_proj(audio_seq) + self.audio_type_embed  # [B, T_a, d_model]
            kv_parts.append(a_proj)
            if audio_mask is not None:
                mask_parts.append(audio_mask)
            else:
                mask_parts.append(torch.zeros(B, audio_seq.shape[1],
                                              dtype=torch.bool, device=audio_seq.device))

        if video_seq is not None:
            B = video_seq.shape[0]
            v_proj = self.video_proj(video_seq) + self.video_type_embed  # [B, T_v, d_model]
            kv_parts.append(v_proj)
            if video_mask is not None:
                mask_parts.append(video_mask)
            else:
                mask_parts.append(torch.zeros(B, video_seq.shape[1],
                                              dtype=torch.bool, device=video_seq.device))

        assert len(kv_parts) > 0, "QFormerBridge requires at least one modality sequence"

        kv = torch.cat(kv_parts, dim=1)        # [B, T_a + T_v, d_model]
        kv_mask = torch.cat(mask_parts, dim=1)  # [B, T_a + T_v]

        # Expand query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_queries, d_model]

        # Apply cross-attention layers
        for layer in self.layers:
            queries = layer(queries, kv, kv_mask=kv_mask)

        # Project to LLM dimension
        return self.output_proj(queries)  # [B, num_queries, output_dim]


# ══════════════════════════════════════════════════════════════
# 2. CrossAttnExpander — legacy lightweight ablation baseline
#    (formerly named QFormerBridge, sits in Fusion layer)
# ══════════════════════════════════════════════════════════════

class CrossAttnExpander(nn.Module):
    """Lightweight cross-attention token expander (ablation baseline).

    Receives a single fused vector from Mixer output and expands it into
    pseudo-tokens via multi-scale projection + learnable query cross-attention.
    Functionally parallel to MSF / SD-MoE in the Fusion layer.

    Interface: same as MSF
        Input:  feature_f  [B, 256]  (from Mixer output)
        Output: pseudo_tokens [B, num_queries, text_in]  (for LLM prompt)
    """

    def __init__(self, input_dim=256, output_dim=4096, num_queries=4,
                 d_model=256, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries

        # ── Input projection: expand feature_f to sequence for cross-attention KV ──
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # Multi-scale expansion: create 3 views of the input at different granularities
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // r),
                nn.GELU(),
                nn.Linear(input_dim // r, d_model),
                nn.LayerNorm(d_model),
            ) for r in [4, 8, 16]
        ])

        # ── Learnable query tokens ──
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, d_model) * 0.02
        )

        # ── Cross-Attention + FFN layers ──
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(QFormerLayer(d_model, num_heads, dropout))

        # ── Output projection to LLM hidden dimension ──
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, feature_f):
        """
        Args:
            feature_f: [B, input_dim] — fused modality feature from Mixer

        Returns:
            pseudo_tokens: [B, num_queries, output_dim] — for LLM input
        """
        B = feature_f.shape[0]

        # Build KV sequence from multi-scale projections: [B, 4, d_model]
        kv_parts = [self.input_proj(feature_f).unsqueeze(1)]  # [B, 1, d_model]
        for proj in self.scale_projs:
            kv_parts.append(proj(feature_f).unsqueeze(1))     # [B, 1, d_model]
        kv = torch.cat(kv_parts, dim=1)  # [B, 4, d_model]

        # Expand query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_queries, d_model]

        # Apply cross-attention layers
        for layer in self.layers:
            queries = layer(queries, kv)

        # Project to LLM dimension
        return self.output_proj(queries)  # [B, num_queries, output_dim]
