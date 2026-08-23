# -*- coding:utf-8 -*-
"""
@author: Ryuk
@contact: jeryuklau@gmail.com
CRN: Convolutional Recurrent Network for speech enhancement.
"""
import torch
import torch.nn as nn
from .common.layers import CausalConvEncoder, CausalConvDecoder


class CrnModel(nn.Module):
    def __init__(self, lstm_hidden_dim, n_lstm_layers):
        super().__init__()
        self.en = CausalConvEncoder(
            channels=[1, 16, 32, 64, 128, 256],
            activation=nn.ELU,
        )
        self.lstm = nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, n_lstm_layers, batch_first=True)
        self.de = CausalConvDecoder(
            channels=[256, 128, 64, 32, 16, 1],
            activation=nn.ELU,
            final_activation=nn.Softplus,
            use_bn_last=True,
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        batch_size, _, seq_len, _ = x.shape
        x, x_list = self.en(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = x.view(batch_size, seq_len, 256, 4)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.de(x, x_list)
        return x


def build_crn(lstm_hidden_dim, n_lstm_layers):
    return CrnModel(lstm_hidden_dim, n_lstm_layers)
