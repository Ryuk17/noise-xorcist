"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""


import torch
import torch.nn.functional as F


def mask_mse_loss(preds, batched_inputs, eps):
    y_ms = batched_inputs['y_ms'].clamp(min=eps)
    x_ms = batched_inputs['x_ms'].clamp(min=eps)
    mask = x_ms / y_ms
    mag_mse = F.mse_loss(preds, mask)
    return mag_mse


def weighted_speech_loss(preds, batched_inputs, alpha):
    vad = batched_inputs['vad']
    y_ms = batched_inputs['y_ms']
    noise_ms = batched_inputs['noise_ms']

    vad_bin = torch.unsqueeze(vad, dim=1).expand_as(y_ms)
    loss_speech = F.mse_loss(y_ms[vad_bin], (preds * y_ms)[vad_bin])
    loss_noise = F.mse_loss(torch.zeros_like(preds), preds * noise_ms)
    loss_val = alpha * loss_speech + (1 - alpha) * loss_noise
    return loss_val