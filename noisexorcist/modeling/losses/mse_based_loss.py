"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""


import torch


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