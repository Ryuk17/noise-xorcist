# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F



logger = logging.getLogger(__name__)


class NegativeSNRLoss(nn.Module):
    """
    Negative Signal-to-Noise Ratio loss.

    Calculates the negative SNR over a predicted output and ground truth
    output pair.

    Args:
        eps (float): Machine epsilon.
    """

    def __init__(self, cfg):
        super().__init__()
        self.eps = cfg["eps"]

    def forward(self, y_pred: torch.tensor, y: torch.tensor):
        """ Calculates the negative SNR loss based on a predicted output and a
        ground truth output.

        Args:
            y_pred (torch.tensor): Predicted tensor containing the denoised
                signal.

            y (torch.tensor): Ground truth tensor containing the clean signal.

        Returns:
            loss (torch.tensor): 1D tensor containing the loss function value
        """
        numerator = torch.sum(torch.square(y), dim=-1, keepdim=True)
        denominator = torch.sum(torch.square(y - y_pred), dim=-1, keepdim=True)
        loss = -10 * torch.log10(numerator / denominator + self.eps)
        # experimental result based on 7 significant digits for torch.float32
        loss[torch.isneginf(loss)] = -140.0
        return torch.mean(loss, dim=0)


class GainMaskBasedNegativeSNRLoss(nn.Module):
    """ Negative Signal-to-Noise Ratio loss for gain mask based networks.

    Calculates the negative SNR over a predicted spectral mask and a complex
    stft output of a noisy speech signal and ground truth clean signal.
    """

    def __init__(self, cfg):
        super().__init__()
        self.window_size = cfg["window_size"]
        self.hop_size = cfg["hop_size"]
        self.eps = cfg["eps"]
        self._window = torch.hann_window(self.window_size)
        self._negative_snr_loss = NegativeSNRLoss(eps=self.eps)

    def istft(self, x_complex: torch.tensor):
        window = self._window.to(x_complex.device)

        istft = torch.istft(x_complex,
                            onesided=True,
                            center=True,
                            n_fft=self.window_size,
                            hop_length=self.hop_size,
                            normalized=False,
                            window=window)

        return istft

    def forward(self, y_pred_mask: torch.tensor, x_complex: torch.tensor,
                y_complex: torch.tensor):
        """
        Calculates the negative SNR over a predicted spectral mask and a complex
        stft output of a noisy speech signal and ground truth clean signal.

        Args:
            y_pred_mask (torch.tensor): Predicted tensor containing the gain mask
                to be applied to the complex stft input x_complex.

            x_complex (torch.tensor): Tensor containing the complex stft of
                the input signal.

            y_complex (torch.tensor): Tensor containing the ground truth complex
                stft of the output signal.

        Returns:
            loss (torch.tensor): 1D tensor containing the loss function value
        """
        y_pred_complex = y_pred_mask.squeeze(1).permute(0, 2, 1) * x_complex
        y_pred = self.istft(y_pred_complex)
        y = self.istft(y_complex)

        return self._negative_snr_loss(y_pred, y)


class SISNRLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, estimate_source, source, EPS =1e-6):
        """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
        Args:
            source: torch tensor, [batch size, sequence length]
            estimate_source: torch tensor, [batch size, sequence length]
        Returns:
            SISNR, [batch size]
        """
        assert source.size() == estimate_source.size()

        # Step 1. Zero-mean norm
        source = source - torch.mean(source, axis = -1, keepdim=True)
        estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)

        # Step 2. SI-SNR
        # s_target = <s', s>s / ||s||^2
        ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
        
        proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
        
        # e_noise = s' - s_target
        noise = estimate_source - proj
        
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
        sisnr = 10 * torch.log10(ratio + EPS)
        
        return sisnr