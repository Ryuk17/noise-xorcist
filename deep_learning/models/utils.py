'''
Author: Ryuk
Date: 2026-02-18 14:34:10
LastEditors: Ryuk
LastEditTime: 2026-02-18 14:34:22
Description: First create
'''

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
from torch.types import Number
def as_complex(x: Tensor):
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Last dimension need to be of length 2 (re + im), but got {x.shape}")
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)


def as_real(x: Tensor):
    if torch.is_complex(x):
        return torch.view_as_real(x)
    return x


class angle_re_im(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, re: Tensor, im: Tensor):
        ctx.save_for_backward(re, im)
        return torch.atan2(im, re)

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tuple[Tensor, Tensor]:
        re, im = ctx.saved_tensors
        grad_inv = grad / (re.square() + im.square()).clamp_min_(1e-10)
        return -im * grad_inv, re * grad_inv


class angle(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))