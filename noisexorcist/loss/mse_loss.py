# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging

import torch
import torch.nn as nn
logger = logging.getLogger(__name__)


class WeightedSpeechDistortionLoss(nn.Module):  # weighted phase magnitude loss
    def __init__(self, cfg):
        super(WeightedSpeechDistortionLoss, self).__init__()
        self.alpha = cfg["ALPHA"]

    def forward(self, inputs, labels):
        x_lps, x_ms, y_ms, noise_ms, VAD = labels
        y_hat = inputs

        VAD_expanded = torch.unsqueeze(VAD, dim=1).expand_as(y_ms)
        loss_speech = self.loss(y_ms[VAD_expanded], (y_hat * y_ms)[VAD_expanded])
        loss_noise = self.loss(torch.zeros_like(y_hat), y_hat * noise_ms)

        loss_val = self.alpha * loss_speech + (1 - self.alpha) * loss_noise
        return loss_val
