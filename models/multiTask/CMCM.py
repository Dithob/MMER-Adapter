# self supervised multimodal multi-task learning network
import math
import os
import sys
import collections
from torch.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.Textmodel import Language_model

__all__ = ['CMCM']

class CMCM(nn.Module):
    def __init__(self, args):
        super(CMCM, self).__init__()
        # text enocding
        self.LLM = Language_model(args)

        # audio and video enocding
        text_in, audio_in, video_in = args.feature_dims[:]
        text_len, audio_len, video_len = args.seq_lens[:]
        
        # Dynamically resolve text_in from LLM hidden_size
        # When feature_dims[0]=0 (placeholder), always use LLM's actual hidden_size
        if hasattr(self.LLM.model, 'config'):
            llm_hidden_size = getattr(self.LLM.model.config, 'hidden_size', None)
            if llm_hidden_size is not None and (text_in == 0 or text_in != llm_hidden_size):
                text_in = llm_hidden_size
                # Write back to args for consistency in logs and downstream
                args.feature_dims = (text_in, audio_in, video_in)

        self.audio_LSTM = TVA_LSTM(audio_in, args.a_lstm_hidden_size, num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_LSTM = TVA_LSTM(video_in, args.v_lstm_hidden_size, num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        self.use_moe_fusion = getattr(args, 'use_moe_fusion', False)
        self.use_gate = getattr(args, 'use_gate', False)
        self.use_moe_lb_loss = getattr(args, 'use_moe_lb_loss', False)
        
        fusion_input_size = 256
        # Shared Text_guide_mixer (used by both MoE and non-MoE paths)
        self.text_guide_mixer = Text_guide_mixer(text_in)

        if self.use_moe_fusion:
            self.pseudo_tokens = args.pseudo_tokens
            self.text_in = text_in
            # Expert 1: Global Emotion — Deep multi-scale fusion
            self.expert1_fusion = mutli_scale_fusion(input_size=fusion_input_size, output_size=text_in, pseudo_tokens=args.pseudo_tokens)
            
            # Expert 2: Micro Emotion — Single-scale bottleneck + residual
            self.expert2_fusion = nn.Sequential(
                nn.Linear(fusion_input_size, 128),
                nn.GELU(),
                nn.Linear(128, fusion_input_size)
            )
            # Shared projector for Expert2 (256 → text_in → pseudo_tokens)
            self.shared_projector = nn.Linear(fusion_input_size, text_in)
            self.shared_token_projector = nn.Linear(1, args.pseudo_tokens)
            
            # Gate: route from shared fusion feature (semantically aligned with experts)
            gate_input_dim = fusion_input_size  # 256
            if self.use_gate:
                gate_input_dim += 1  # + bias_score
            self.gate = nn.Linear(gate_input_dim, 2)
        else:
            self.mutli_scale_fusion = mutli_scale_fusion(input_size=fusion_input_size, output_size=text_in, pseudo_tokens=args.pseudo_tokens)

    def _compute_moe_lb_loss(self, gate_weights):
        """Load-balance loss to prevent routing collapse (ref: MMGAT_EMO)"""
        expert_load = torch.mean(gate_weights, dim=0)  # [num_experts]
        lb_loss = -torch.sum(expert_load * torch.log(expert_load + 1e-10))
        return lb_loss

    def _moe_forward(self, audio_h, video_h, feature_f):
        """MoE forward: Gate → Expert → WeightedSum"""
        # 1. Gate routing (from shared fusion feature)
        gate_input = feature_f
        if self.use_gate:
            bias_score = F.cosine_similarity(audio_h, video_h, dim=-1).unsqueeze(-1)
            gate_input = torch.cat([feature_f, bias_score], dim=-1)
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)
        
        # 2. Expert 1: Global Emotion (multi-scale deep fusion)
        e1_out = self.expert1_fusion(feature_f)
        
        # 3. Expert 2: Micro Emotion (single-scale + residual + shared projector)
        e2_refined = self.expert2_fusion(feature_f) + feature_f  # residual
        e2_projected = self.shared_projector(e2_refined)
        e2_out = self.shared_token_projector(e2_projected.unsqueeze(2))
        e2_out = e2_out.permute(0, 2, 1)  # [B, pseudo_tokens, text_in]
        
        # 4. Weighted fusion
        fusion_h = gate_weights[:, 0].unsqueeze(1).unsqueeze(2) * e1_out + \
                   gate_weights[:, 1].unsqueeze(1).unsqueeze(2) * e2_out
        
        return fusion_h, gate_weights

    def forward(self, labels, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text
        text = self.LLM.text_embedding(text[:,0,:].long())

        video_h = self.video_LSTM(video, video_len)
        audio_h = self.audio_LSTM(audio, audio_len)

        # Shared mixer (computed once)
        feature_f = self.text_guide_mixer(audio_h, video_h, text)

        if self.use_moe_fusion:
            fusion_h, gate_weights = self._moe_forward(audio_h, video_h, feature_f)
        else:
            fusion_h = self.mutli_scale_fusion(feature_f)

        LLM_input = torch.cat([fusion_h, text], dim=1)
        LLM_output = self.LLM(LLM_input, labels)

        res = {
            'Loss': LLM_output.loss,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': feature_f,
        }

        # Optional: MoE load-balance loss
        if self.use_moe_fusion and self.use_moe_lb_loss and self.training:
            lb_loss = self._compute_moe_lb_loss(gate_weights)
            res['MoE_LB_Loss'] = lb_loss * 0.01

        return res

    def generate(self, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text
        text = self.LLM.text_embedding(text[:,0,:].long())

        audio_h = self.audio_LSTM(audio, audio_len)
        video_h = self.video_LSTM(video, video_len)

        # Shared mixer (computed once)
        feature_f = self.text_guide_mixer(audio_h, video_h, text)

        if self.use_moe_fusion:
            fusion_h, _ = self._moe_forward(audio_h, video_h, feature_f)
        else:
            fusion_h = self.mutli_scale_fusion(feature_f)

        LLM_input = torch.cat([fusion_h, text], dim=1)
        LLM_output = self.LLM.generate(LLM_input)

        return LLM_output



class TVA_LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TVA_LSTM, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 256)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False) #这里把length.to cpu是因为pytorch版本问题
        # _, (final_states, _) = self.rnn(packed_sequence)
        # h = self.dropout(final_states[-1])
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze(0))
        h = self.linear(h)
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
    def __init__(self, input_size, output_size, pseudo_tokens = 4):
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

        self.integrating = Integrating(scales = 3)
        self.multi_scale_projector =  nn.Linear(multi_scale_hidden, output_size)
        self.projector = nn.Linear(1, pseudo_tokens)

    def forward(self,x):
        # 增加样本复制，将单一样本复制一份,避免最后一个batch只有一个数据时的报错
        if x.dim() == 1:
            x = x.unsqueeze(0)
        #compute different scale experts outputs
        scale1 = self.scale1(x)
        scale2 = self.scale2(x)
        scale3 = self.scale3(x)


        # Calculate the expert outputs
        multi_scale_stack = torch.stack([scale1, scale2, scale3], dim=2)
        multi_scale_integrating =  self.integrating(multi_scale_stack)

        multi_scale = self.multi_scale_projector(multi_scale_integrating)
        output = self.projector(multi_scale.unsqueeze(2))
        return output.permute(0, 2, 1)  #[batch,seq_len,hidden_siez]

# Define the gating model
class Integrating(nn.Module):
    def __init__(self,  scales):
        super(Integrating, self).__init__()

    # Layers
        self.Integrating_layer = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(1, scales), stride=1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.Integrating_layer(x)
        x = x.squeeze((1, 3))
        return x
