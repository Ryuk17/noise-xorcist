'''
Author: Ryuk
Date: 2026-02-18 12:45:19
LastEditors: Ryuk
LastEditTime: 2026-02-18 14:11:30
Description: First create
'''


import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import replace_denormals


logger = logging.getLogger(__name__)


class WeightedSpeechDistortionLoss(nn.Module):
    def __init__(self, cfg):
        super(WeightedSpeechDistortionLoss, self).__init__()
        self.alpha = cfg["ALPHA"]

    def forward(self, inputs, data):
        x_lps, x_ms, y_ms, noise_ms, VAD = data["x_lps"], data["x_ms"], data["y_ms"], data["noise_ms"], data["VAD"]
        y_hat = inputs

        VAD_expanded = torch.unsqueeze(VAD, dim=1).expand_as(y_ms)
        loss_speech = F.mse_loss(y_ms[VAD_expanded], (y_hat * y_ms)[VAD_expanded])
        loss_noise = F.mse_loss(torch.zeros_like(y_hat), y_hat * noise_ms)

        loss_val = self.alpha * loss_speech + (1 - self.alpha) * loss_noise
        return loss_val


class ComplexCompressedMSELoss(nn.Module):
    """ Complex Compressed Mean Square Error Loss implemented as shown in
    section two of:
        https://arxiv.org/pdf/2101.09249.pdf

    c_ (float): Compression factor.
    lambda_ (float): Weighting factor.
    eps (float): Machine epsilon.
    """

    def __init__(self, cfg):
        super().__init__()
        self.c_ = cfg["c"]
        self.lambda_ = cfg["lambda"]
        self.eps = cfg["eps"]

    def forward(self, y_pred_mask: torch.tensor, x_complex: torch.tensor,
                y_complex: torch.tensor):
        # clean denormals
        y_complex = replace_denormals(torch.real(y_complex)) + \
                    1j * torch.imag(y_complex)

        # get target magnitude and phase
        y_mag = torch.abs(y_complex)
        y_phase = torch.angle(y_complex)

        # predicted complex stft
        y_pred_mask = y_pred_mask.squeeze(1).permute(0, 2, 1)
        y_pred_complex = y_pred_mask.type(torch.complex64) * x_complex

        # clean denormals
        y_pred_complex = replace_denormals(torch.real(y_pred_complex)) + \
                         1j * torch.imag(y_pred_complex)

        # get predicted magnitude annd phase
        y_pred_mag = torch.abs(y_pred_complex)
        y_pred_phase = torch.angle(y_pred_complex)

        # target complex exponential
        y_complex_exp = (y_mag ** self.c_).type(torch.complex64) * \
                        torch.exp(1j * y_phase.type(torch.complex64))

        # predicted complex exponential
        y_pred_complex_exp = (y_pred_mag ** self.c_).type(torch.complex64) * \
                             torch.exp(1j * y_pred_phase.type(torch.complex64))

        # magnitude only loss component
        mag_loss = torch.abs(y_mag ** self.c_ - y_pred_mag ** self.c_) ** 2
        mag_loss = torch.sum(mag_loss, dim=[1, 2])

        # complex loss component
        complex_loss = torch.abs(y_complex_exp - y_pred_complex_exp) ** 2
        complex_loss = torch.sum(complex_loss, dim=[1, 2])

        # blend both loss components
        loss = (1 - self.lambda_) * mag_loss + (self.lambda_) * complex_loss

        # returns the mean blended loss of the batch
        return torch.mean(loss)


class STFTLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_len=120, win_len=600, window="hann_window"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.register_buffer("window", getattr(torch, window)(win_len))

    def loss_spectral_convergence(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

    def loss_log_magnitude(self, x_mag, y_mag):
        return torch.nn.functional.l1_loss(torch.log(y_mag), torch.log(x_mag))

    def forward(self, x, y):
        """x, y: (B, T), in time domain"""
        x = torch.stft(x, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        y = torch.stft(y, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        x_mag = torch.abs(x).clamp(1e-8)
        y_mag = torch.abs(y).clamp(1e-8)
        
        sc_loss = self.loss_spectral_convergence(x_mag, y_mag)
        mag_loss = self.loss_log_magnitude(x_mag, y_mag)
        loss = sc_loss + mag_loss

        return loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[240, 120, 50],
        win_lengths=[1200, 600, 240],
        window="hann_window",
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, hs, wl, window)]

    def forward(self, x, y):
        loss = 0.0
        for f in self.stft_losses:
            loss += f(x, y)
        loss /= len(self.stft_losses)
        return loss

