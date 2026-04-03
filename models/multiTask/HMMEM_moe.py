import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GlobalMoE', 'LocalMoE']


class GlobalMoE(nn.Module):
    """Global Emotion MoE with heterogeneous inductive-bias experts.

    Expert 1 — Bilinear Interaction: captures second-order feature correlations.
    Expert 2 — SE Channel Attention: recalibrates feature channels adaptively.
    Expert 3 — Linear Residual: simple linear transform + skip connection (regularization anchor).

    All experts: [B, D] → [B, D].  Gate routes among them with softmax weights.
    """

    def __init__(self, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.num_experts = 3

        # ── Expert 1: Bilinear Interaction ──
        half = input_size // 2
        self.bilinear = nn.Bilinear(half, half, input_size)

        # ── Expert 2: SE Channel Attention ──
        self.se = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, input_size),
            nn.Sigmoid()
        )

        # ── Expert 3: Linear Residual ──
        self.linear_expert = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU()
        )

        # ── Gate ──
        self.gate = nn.Linear(input_size, self.num_experts)

    def forward(self, x):
        """
        Args:  x [B, D]
        Returns:
            output        [B, D]
            gate_weights  [B, 3]
            expert_outputs list of [B, D]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Expert 1: Bilinear
        x1, x2 = x.chunk(2, dim=-1)       # each [B, D/2]
        e1 = self.bilinear(x1, x2)         # [B, D]

        # Expert 2: SE channel attention
        channel_w = self.se(x)             # [B, D] sigmoid weights
        e2 = x * channel_w                 # [B, D]

        # Expert 3: Linear + residual
        e3 = self.linear_expert(x) + x     # [B, D]

        expert_outputs = [e1, e2, e3]

        # Gate routing
        gate_weights = F.softmax(self.gate(x), dim=-1)  # [B, 3]
        output = (gate_weights[:, 0:1] * e1 +
                  gate_weights[:, 1:2] * e2 +
                  gate_weights[:, 2:3] * e3)

        return output, gate_weights, expert_outputs


class LocalMoE(nn.Module):
    """Local Emotion MoE: lightweight bottleneck experts.
    Designed to receive raw modality features (cat(audio_h, video_h) projected to D).
    """

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
        Args:  x [B, D]
        Returns:
            output        [B, D]
            gate_weights  [B, num_experts]
            expert_outputs list of [B, D]
        """
        gate_weights = F.softmax(self.gate(x), dim=-1)
        expert_outputs = [expert(x) for expert in self.experts]

        output = torch.zeros_like(expert_outputs[0])
        for i, e_out in enumerate(expert_outputs):
            output = output + gate_weights[:, i:i+1] * e_out

        return output, gate_weights, expert_outputs
