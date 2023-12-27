"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import torch
import ci_sdr


def negative_snr_loss(ref, inf, eps):
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        noise = inf - ref
        snr = 20 * (
            torch.log10(torch.norm(ref, p=2, dim=1).clamp(min=eps))
            - torch.log10(torch.norm(noise, p=2, dim=1).clamp(min=eps))
        )
        return -snr


def ci_sdr_loss(ref, inf, filter_length=512):
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        ci_sdr = ci_sdr.pt.ci_sdr_loss(inf, ref, compute_permutation=False, filter_length=filter_length)
        return ci_sdr
