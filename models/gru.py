"""
@FileName: gru.py
@Description: Implement gru
@Author: Ryuk
@CreateDate: 2023/06/20
@LastEditTime: 2023/06/20
@LastEditors: Please set LastEditors
@Version: v0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GruNet(nn.Module):
    def __init__(self, cfg):
        super(GruNet, self).__init__()
        self.n_fft = cfg['DATA']['N_FFT']
        self.n_frequency_bins = self.n_fft // 2 + 1
        self.n_gru_layers = cfg['MODEL']['N_GRU_LAYERS']
        self.gru_dropout = cfg['MODEL']['GRU_DROPOUT']
        self.alpha = cfg['MODEL']['ALPHA']

        self.gru = nn.GRU(input_size=self.n_frequency_bins, hidden_size=self.n_frequency_bins,
                          num_layers=self.n_gru_layers,
                          batch_first=True, dropout=self.gru_dropout)
        self.dense = nn.Linear(in_features=self.n_frequency_bins, out_features=self.n_frequency_bins)

    def forward(self, x):
        x = x.permute(0, 2, 1)              # (batch_size, time, n_frequency_bins)
        x, _ = self.gru(x)                  # (batch_size, time, n_frequency_bins)
        x = torch.sigmoid(self.dense(x))    # (batch_size, time, frequency_bins)
        x = x.permute(0, 2, 1)              # (batch_size, frequency_bins, time)
        return x
