"""
AIO -- All Model in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal


from models.multiTask import *

__all__ = ['AMIO']

MODEL_MAP = {
    'hmmem': HMMEM
}

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        lastModel = MODEL_MAP[args.modelName]
        self.Model = lastModel(args)

    def forward(self, labels_m, text_x, audio_x, video_x, context_text=None):
        return self.Model(labels_m, text_x, audio_x, video_x, context_text=context_text)

    def generate(self, text_x, audio_x, video_x, context_text=None):
        # Returns (predictions, feature_f) tuple
        return self.Model.generate(text_x, audio_x, video_x, context_text=context_text)
