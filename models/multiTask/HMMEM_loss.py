import torch
import torch.nn as nn

__all__ = ['DiffLoss', 'LightweightCrossCPC']

class DiffLoss(nn.Module):
    """Orthogonality loss: encourages two representations to be different (complementary).
    Ref: MMGAT_EMO/model_utils.py"""
    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        # Zero mean
        input1 = input1 - torch.mean(input1, dim=0, keepdim=True)
        input2 = input2 - torch.mean(input2, dim=0, keepdim=True)
        # L2 normalize
        input1_l2 = input1 / (torch.norm(input1, p=2, dim=1, keepdim=True) + 1e-6)
        input2_l2 = input2 / (torch.norm(input2, p=2, dim=1, keepdim=True) + 1e-6)
        # Inverse distance → smaller distance = higher loss
        diff_loss = 1.0 / (torch.mean(torch.norm(input1_l2 - input2_l2, p=2, dim=1)) + 1e-6)
        return diff_loss


class LightweightCrossCPC(nn.Module):
    """Lightweight Cross-modal CPC (Contrastive Predictive Coding).
    Contrasts text↔audio or text↔video using temporal features.
    Ref: MMGAT_EMO/CPC.py (Cross_CPC), adapted for MMER-Adapter."""
    def __init__(self, text_dim, other_dim, nce_hidden_dim=32, n_prediction_steps=2, min_start_steps=3):
        super().__init__()
        self.nce_hidden_dim = nce_hidden_dim
        self.n_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.lsoftmax = nn.LogSoftmax(dim=1)

        # Internal projections to unified nce_hidden_dim
        self.text_proj = nn.Linear(text_dim, nce_hidden_dim)
        self.other_proj = nn.Linear(other_dim, nce_hidden_dim)

        # Lightweight autoregressive LSTMs (1 layer)
        self.text_ar = nn.LSTM(nce_hidden_dim, nce_hidden_dim, 1, batch_first=True)
        self.other_ar = nn.LSTM(nce_hidden_dim, nce_hidden_dim, 1, batch_first=True)

        # Prediction heads
        self.text_predictors = nn.ModuleList([
            nn.Linear(nce_hidden_dim, nce_hidden_dim) for _ in range(n_prediction_steps)
        ])
        self.other_predictors = nn.ModuleList([
            nn.Linear(nce_hidden_dim, nce_hidden_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, text_seq, other_seq):
        """
        Args:
            text_seq:  [B, T_text, text_dim]   — LLM text embeddings
            other_seq: [B, T_other, other_dim] — LSTM temporal output (audio or video)
        Returns:
            nce_loss: scalar
        """
        # Project to unified space
        text_vq = self.text_proj(text_seq)     # [B, T_text, nce_hidden_dim]
        other_vq = self.other_proj(other_seq)  # [B, T_other, nce_hidden_dim]

        # Use the shorter sequence length for both
        T = min(text_vq.size(1), other_vq.size(1))
        text_vq = text_vq[:, :T, :]
        other_vq = other_vq[:, :T, :]

        batch_dim = text_vq.size(0)

        # Need at least min_start_steps + n_prediction_steps time steps
        if T < self.min_start_steps + self.n_steps + 1:
            return torch.tensor(0.0, device=text_vq.device, requires_grad=True)

        t_samples = (torch.randint(T - self.n_steps - self.min_start_steps, size=(1,)) + self.min_start_steps).long()

        nce = 0.0
        # Encode future samples
        text_encode = torch.stack([text_vq[:, t_samples + i, :].squeeze(1) for i in range(self.n_steps)])
        other_encode = torch.stack([other_vq[:, t_samples + i, :].squeeze(1) for i in range(self.n_steps)])

        # Autoregressive context
        text_forward = text_vq[:, :t_samples + 1, :]
        other_forward = other_vq[:, :t_samples + 1, :]

        h_text = (torch.zeros(1, batch_dim, self.nce_hidden_dim, device=text_vq.device),
                  torch.zeros(1, batch_dim, self.nce_hidden_dim, device=text_vq.device))
        h_other = (torch.zeros(1, batch_dim, self.nce_hidden_dim, device=other_vq.device),
                   torch.zeros(1, batch_dim, self.nce_hidden_dim, device=other_vq.device))

        text_context, _ = self.text_ar(text_forward, h_text)
        other_context, _ = self.other_ar(other_forward, h_other)

        text_context = text_context[:, -1, :]    # [B, nce_hidden_dim]
        other_context = other_context[:, -1, :]  # [B, nce_hidden_dim]

        # Predict and compute NCE
        for i in range(self.n_steps):
            text_pred = self.text_predictors[i](text_context)
            other_pred = self.other_predictors[i](other_context)

            # Cross-modal: text predicts other's future, other predicts text's future
            total1 = torch.mm(text_encode[i], other_pred.t())
            total2 = torch.mm(other_encode[i], text_pred.t())
            nce += torch.sum(torch.diag(self.lsoftmax(total1)))
            nce += torch.sum(torch.diag(self.lsoftmax(total2)))

        nce /= -1.0 * batch_dim * self.n_steps
        return 0.05 * nce
