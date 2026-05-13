import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['OriginGlobalMoE', 'OriginLocalMoE', 'GlobalSemanticMoE', 'LocalDetailMoE']


# ═══════════════════════════════════════════════════════════════
# Original MoE (preserved for ablation — renamed from GlobalMoE/LocalMoE)
# ═══════════════════════════════════════════════════════════════

class OriginGlobalMoE(nn.Module):
    """Original Global Emotion MoE with heterogeneous inductive-bias experts.

    Expert 1 — Bilinear Interaction: captures second-order feature correlations.
    Expert 2 — SE Channel Attention: recalibrates feature channels adaptively.
    Expert 3 — Linear Residual: simple linear transform + skip connection.
    """

    def __init__(self, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.num_experts = 3

        half = input_size // 2
        rank = 32
        self.bilinear_proj1 = nn.Linear(half, rank)
        self.bilinear_proj2 = nn.Linear(half, rank)
        self.bilinear_out = nn.Linear(rank, input_size)

        self.se = nn.Sequential(
            nn.Linear(input_size, input_size // 4), nn.ReLU(),
            nn.Linear(input_size // 4, input_size), nn.Sigmoid()
        )

        self.linear_expert = nn.Sequential(
            nn.Linear(input_size, input_size), nn.GELU()
        )

        self.gate = nn.Linear(input_size, self.num_experts)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x1, x2 = x.chunk(2, dim=-1)
        e1 = self.bilinear_out(self.bilinear_proj1(x1) * self.bilinear_proj2(x2))
        e2 = x * self.se(x)
        e3 = self.linear_expert(x) + x

        expert_outputs = [e1, e2, e3]
        gate_weights = F.softmax(self.gate(x), dim=-1)
        output = (gate_weights[:, 0:1] * e1 +
                  gate_weights[:, 1:2] * e2 +
                  gate_weights[:, 2:3] * e3)
        return output, gate_weights, expert_outputs


class OriginLocalMoE(nn.Module):
    """Original Local Emotion MoE: lightweight bottleneck experts."""

    def __init__(self, input_size=256, num_experts=3, bottleneck_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, bottleneck_dim), nn.GELU(),
                nn.Linear(bottleneck_dim, input_size)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_size, num_experts)

    def forward(self, x):
        gate_weights = F.softmax(self.gate(x), dim=-1)
        expert_outputs = [expert(x) for expert in self.experts]
        output = torch.zeros_like(expert_outputs[0])
        for i, e_out in enumerate(expert_outputs):
            output = output + gate_weights[:, i:i+1] * e_out
        return output, gate_weights, expert_outputs


# ═══════════════════════════════════════════════════════════════
# SD-MoE: Semantic-Decomposed Mixture of Experts (improved v3)
# ═══════════════════════════════════════════════════════════════

class GlobalSemanticMoE(nn.Module):
    """Global Semantic MoE: processes coarse-grained emotion semantics (z_shared).

    Heterogeneous experts extract emotion patterns from shared semantic features.
    """

    def __init__(self, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.num_experts = 3

        # Expert 1: Low-Rank Bilinear (second-order correlation)
        half, rank = input_size // 2, 32
        self.e1_proj1 = nn.Linear(half, rank)
        self.e1_proj2 = nn.Linear(half, rank)
        self.e1_out = nn.Linear(rank, input_size)

        # Expert 2: SE Channel Attention (emotional salience)
        self.e2_se = nn.Sequential(
            nn.Linear(input_size, input_size // 4), nn.ReLU(),
            nn.Linear(input_size // 4, input_size), nn.Sigmoid(),
        )

        # Expert 3: Linear Residual (regularisation anchor)
        self.e3_linear = nn.Sequential(nn.Linear(input_size, input_size), nn.GELU())

        self.gate = nn.Linear(input_size, self.num_experts)

    def forward(self, z_shared):
        if z_shared.dim() == 1:
            z_shared = z_shared.unsqueeze(0)
        x1, x2 = z_shared.chunk(2, dim=-1)
        e1 = self.e1_out(self.e1_proj1(x1) * self.e1_proj2(x2))
        e2 = z_shared * self.e2_se(z_shared)
        e3 = self.e3_linear(z_shared) + z_shared

        expert_outputs = [e1, e2, e3]
        gw = F.softmax(self.gate(z_shared), dim=-1)
        output = gw[:, 0:1] * e1 + gw[:, 1:2] * e2 + gw[:, 2:3] * e3
        return output, gw, expert_outputs


class LocalDetailMoE(nn.Module):
    """Local Detail MoE: processes fine-grained emotion residuals.

    Uses *descending bottleneck* dimensions across experts so that each
    expert captures a different granularity of residual information.
    """

    def __init__(self, input_size=256, num_experts=3, bottleneck_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, max(bottleneck_dim // (i + 1), 16)),
                nn.GELU(),
                nn.Linear(max(bottleneck_dim // (i + 1), 16), input_size),
            ) for i in range(num_experts)
        ])
        self.gate = nn.Linear(input_size, num_experts)

    def forward(self, residual):
        gw = F.softmax(self.gate(residual), dim=-1)
        expert_outputs = [expert(residual) for expert in self.experts]
        output = sum(gw[:, i:i+1] * expert_outputs[i] for i in range(self.num_experts))
        return output, gw, expert_outputs
