# -*- coding:utf-8 -*-
"""
@author:：Ryuk
@contact: jeryuklau@gmail.com
"""

import torch
import logging

logger = logging.getLogger("noisexorcist")


def select_inputs(cfg, data):
    try:
        if cfg["DATA"]["FEATURE"] == "Spectrum":
            return data["x_lps"]
    except KeyError as e:
        logger.error(f"Invalid feature named {cfg['DATA']['FEATURE']}")
        raise e


def restore_waveform(cfg, data, model_outputs):
    try:
        if cfg["DATA"]["FEATURE"] == "Spectrum":
            noisy_ms_hat = data["x_ms"] * model_outputs
            clean_stft = data["y_stft"]

            denoised_stft_hat = torch.stack([noisy_ms_hat * torch.cos(torch.angle(data["x_stft"])),
                                      noisy_ms_hat * torch.sin(torch.angle(data["x_stft"]))], dim=-1)

            window = build_window(cfg["DATA"]["WIN_TYPE"], cfg["DATA"]["N_FFT"])
            clean_waveform = torch.istft(
                clean_stft, cfg["DATA"]["N_FFT"], hop_length=cfg["DATA"]["HOP_LEN"],
                win_length=cfg["DATA"]["N_FFT"], window=window)

            denoised_waveform = torch.istft(
                denoised_stft_hat, cfg["DATA"]["N_FFT"], hop_length=cfg["DATA"]["HOP_LEN"],
                win_length=cfg["DATA"]["N_FFT"], window=window, length=clean_waveform.shape[-1])
            return denoised_waveform, clean_waveform
    except KeyError as e:
        logger.error(f"Invalid feature named {cfg['DATA']['FEATURE']}")
        raise e


def build_window(win_type, win_len):
    if win_type == "hamming":
        return torch.hamming_window(win_len)
    elif win_type == "hanning":
        return torch.hanning_window(win_len)
    else:
        raise NotImplementedError
