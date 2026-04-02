import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GlobalMoE', 'LocalMoE']

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
