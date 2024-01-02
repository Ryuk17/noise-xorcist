"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import torch
import ci_sdr


def negative_snr_loss(preds, batched_inputs, eps):
        ref = ...
        inf = batched_inputs['clean_waveform']
        assert preds.shape == inf.shape, (preds.shape, inf.shape)
        noise = inf - preds
        snr_loss = 20 * (
            torch.log10(torch.norm(preds, p=2, dim=1).clamp(min=eps))
            - torch.log10(torch.norm(noise, p=2, dim=1).clamp(min=eps))
        )
        return -snr_loss


def ci_sdr_loss(ref, inf, filter_length=512):
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        ci_sdr_loss = ci_sdr.pt.ci_sdr_loss(inf, ref, compute_permutation=False, filter_length=filter_length)
        return ci_sdr_loss
