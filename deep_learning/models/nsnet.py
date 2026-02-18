# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# fix random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


class NSNetModel(nn.Module):
    def __init__(self, input_dim, n_gru_layers, gru_dropout):
        super(NSNetModel, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=input_dim, num_layers=n_gru_layers,
                          batch_first=True, dropout=gru_dropout)
        self.dense = nn.Linear(in_features=input_dim, out_features=input_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, time, n_frequency_bins)
        x, _ = self.gru(x)  # (batch_size, time, n_frequency_bins)
        x = torch.sigmoid(self.dense(x))  # (batch_size, time, frequency_bins)
        x = x.permute(0, 2, 1)  # (batch_size, frequency_bins, time)

        return x


def build_nsnet(input_dim, n_gru_layers, gru_dropout):
    model = NSNetModel(input_dim, n_gru_layers, gru_dropout)
    return model