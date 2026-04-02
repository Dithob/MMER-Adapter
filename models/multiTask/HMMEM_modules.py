import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['TVA_LSTM', 'Text_guide_mixer', 'Lightweight_mixer', 'mutli_scale_fusion', 'Integrating']

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
