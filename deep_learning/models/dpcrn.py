# -*- coding:utf-8 -*-
"""
@author: Ryuk
@contact: jeryuklau@gmail.com
DPCRN: Dual-Path Convolutional Recurrent Network for speech enhancement.
"""
import torch
import torch.nn as nn
from .common.layers import CausalConvEncoder, CausalConvDecoder, complex_ratio_mask


class DPRNN(nn.Module):
    def __init__(self, in_features=128, out_features=128):
        super().__init__()

        self.intra_rnn = nn.LSTM(in_features, 64, 2, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(in_features, out_features)
        self.inter_rnn = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=False)
        self.inter_fc = nn.Linear(in_features, out_features)

        self.ln1 = nn.LayerNorm([4, out_features])
        self.ln2 = nn.LayerNorm([4, out_features])

    def forward(self, x):
        out = x  # (B,C,T,F)
        # intra
        x = x.permute(0, 2, 3, 1).contiguous()
        batch_size, chan_len, seq_len, freq_len = out.shape
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(-1, freq_len, chan_len)
        out, _ = self.intra_rnn(out)
        out = self.intra_fc(out)
        out = out.view(batch_size, -1, freq_len, chan_len)
        out = self.ln1(out)
        intra_out = out + x  # (B,T,F,C)

        # inter
        out = intra_out.permute(0, 2, 1, 3).contiguous()
        out = out.view(-1, seq_len, chan_len)
        out, _ = self.inter_rnn(out)
        out = self.inter_fc(out)
        out = out.view(batch_size, -1, seq_len, chan_len)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = self.ln2(out)
        out = out + intra_out  # (B,T,F,C)

        out = out.permute(0, 3, 1, 2).contiguous()
        return out


class DpcrnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.en = CausalConvEncoder(
            channels=[2, 32, 32, 32, 64, 128],
            activation=nn.PReLU,
        )
        self.dprnn = DPRNN()
        self.de = CausalConvDecoder(
            channels=[128, 64, 32, 32, 32, 2],
            activation=nn.PReLU,
            final_activation=None,
            use_bn_last=False,
        )

    def forward(self, inpt):
        x = inpt
        x, x_list = self.en(x)  # (B,C,T,F)

        x = self.dprnn(x)
        x = self.dprnn(x)

        mask = self.de(x, x_list)

        return complex_ratio_mask(mask, inpt)


def build_dprn():
    return DpcrnModel()
