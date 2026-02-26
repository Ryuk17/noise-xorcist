'''
Author: Ryuk
Date: 2026-02-26 23:35:10
LastEditors: Ryuk
LastEditTime: 2026-02-26 23:36:32
Description: First create
'''

import torch.nn as nn

class BaseSELoss(nn.Module):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__()
        self.weight = weight

    def forward(self, y_pred, y_true, **kwargs):
        raise NotImplementedError
