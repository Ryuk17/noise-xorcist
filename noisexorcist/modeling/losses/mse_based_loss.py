"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""


import torch
import torch.nn.functional as F


def mag_mse_loss(y, y_hat):
    mag_mse = F.mse_loss(y, y_hat)
    return mag_mse


def weighted_mse_loss(y_ms, noise_ms, vad, y_hat, alpha):
    vad_bin = torch.unsqueeze(vad, dim=1).expand_as(y_ms)
    loss_speech = F.mse_loss(y_ms[vad_bin], (y_hat * y_ms)[vad_bin])
    loss_noise = F.mse_loss(torch.zeros_like(y_hat), y_hat * noise_ms)
    loss_val = alpha * loss_speech + (1 - alpha) * loss_noise
    return loss_val