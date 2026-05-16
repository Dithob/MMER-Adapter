"""Lightweight QFormer Bridge — modality-to-LLM pseudo-token generator.

Replaces mutli_scale_fusion (MSF) when --use_qformer is enabled.
Uses cross-attention with learnable query tokens to selectively extract
information from the fused modality feature, producing pseudo-tokens
aligned with the LLM embedding space.

No external BERT weights required — fully trainable from scratch.

Interface: same as MSF
    Input:  feature_f  [B, 256]  (from Mixer output)
    Output: pseudo_tokens [B, num_queries, text_in]  (for LLM prompt)
"""

import torch
import torch.nn as nn


class QFormerBridge(nn.Module):
    """Lightweight QFormer-style modality bridge.

    Architecture:
        1. Input projection: [B, 256] → [B, 1, d_model] (or multi-token via optional expansion)
        2. Learnable query tokens: [1, num_queries, d_model]
        3. N layers of: Cross-Attention(Q=queries, KV=projected_input) + FFN
        4. Output projection: [B, num_queries, d_model] → [B, num_queries, output_dim]

    Args:
        input_dim:    dimension of input feature (default: 256, from Mixer)
        output_dim:   dimension of output pseudo-tokens (default: text_in / LLM hidden)
        num_queries:  number of learnable query tokens = pseudo_tokens (default: 4)
        d_model:      internal transformer dimension (default: 256)
        num_layers:   number of cross-attention layers (default: 2)
        num_heads:    number of attention heads (default: 4)
        dropout:      dropout rate (default: 0.1)
    """

    def __init__(self, input_dim=256, output_dim=4096, num_queries=4,
                 d_model=256, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries

        # ── Input projection: expand feature_f to sequence for cross-attention KV ──
        # Project to d_model and expand to a short sequence (3 tokens via multi-scale)
        # to give cross-attention more information to attend to
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
        # (1 direct projection + 3 multi-scale views)
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


class QFormerLayer(nn.Module):
    """Single QFormer layer: Cross-Attention + FFN with residual + LayerNorm."""

    def __init__(self, d_model=256, num_heads=4, dropout=0.1):
        super().__init__()

        # Cross-attention: queries attend to KV (projected input features)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention among query tokens (allows inter-token communication)
        self.self_attn = nn.MultiheadAttention(
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

    def forward(self, queries, kv):
        """
        Args:
            queries: [B, num_queries, d_model]
            kv:      [B, kv_len, d_model]
        Returns:
            queries: [B, num_queries, d_model] (refined)
        """
        # Self-attention among queries
        q_sa, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.norm1(queries + q_sa)

        # Cross-attention: queries attend to input features
        q_ca, _ = self.cross_attn(queries, kv, kv, need_weights=False)
        queries = self.norm2(queries + q_ca)

        # FFN
        queries = self.norm3(queries + self.ffn(queries))

        return queries
