"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""


import torch
import torch.nn.functional as F


def mse_loss(ref, inf):
    """Time-domain MSE loss forward.

    Args:
        ref: (Batch, T) or (Batch, T, C)
        inf: (Batch, T) or (Batch, T, C)
    Returns:
        loss: (Batch,)
    """
    assert ref.shape == inf.shape, (ref.shape, inf.shape)

    mseloss = (ref - inf).pow(2)
    if ref.dim() == 3:
        mseloss = mseloss.mean(dim=[1, 2])
    elif ref.dim() == 2:
        mseloss = mseloss.mean(dim=1)
    else:
        raise ValueError(
            "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
        )
    return mseloss


def weighted_mse_loss(y_ms, noise_ms, vad, y_hat, alpha):
    vad_bin = torch.unsqueeze(vad, dim=1).expand_as(y_ms)
    loss_speech = F.mse_loss(y_ms[vad_bin], (y_hat * y_ms)[vad_bin])
    loss_noise = F.mse_loss(torch.zeros_like(y_hat), y_hat * noise_ms)
    loss_val = alpha * loss_speech + (1 - alpha) * loss_noise
    return loss_val