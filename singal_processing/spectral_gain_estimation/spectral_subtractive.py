'''
Author: Ryuk
Date: 2026-02-15 17:08:57
LastEditors: Ryuk
LastEditTime: 2026-02-15 17:22:49
Description: First create
'''

from base import BaseSpectralGainEstimator

import numpy as np

class BeroutiGainEstimator(BaseSpectralGainEstimator):
    def __init__(self, n_fft, alpha=2.0, floor=0.002):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha  # 1.0 为幅值减，2.0 为功率减
        self.floor = floor

    def compute_gain(self, frame_psd, noise_psd):
        """
        基于 Berouti 过减法的增益计算
        """
        # 计算分段信噪比 (Segmental SNR)
        snr_seg = 10 * np.log10(np.sum(frame_psd) / (np.sum(noise_psd) + 1e-12))
        
        # 计算过减因子 beta
        if self.alpha == 1.0:
            beta = np.clip(3.0 - snr_seg * 2.0 / 20.0, 1.0, 4.0)
        else:
            beta = np.clip(4.0 - snr_seg * 3.0 / 20.0, 1.0, 5.0)

        # 核心谱减公式： (Signal^alpha - beta * Noise^alpha) / Signal^alpha
        # 这里直接计算增益 G = (Enhanced_PSD / Noisy_PSD)^(1/alpha)
        noise_part = beta * (noise_psd ** (self.alpha / 2.0))
        signal_part = frame_psd ** (self.alpha / 2.0)
        
        # 减法处理
        sub_res = signal_part - noise_part
        
        # 设置谱下限 (Spectral Floor)
        threshold = self.floor * noise_part
        sub_res = np.where(sub_res < threshold, threshold, sub_res)
        
        # 计算增益因子 G = 估计幅值 / 带噪幅值
        gain = (sub_res / (signal_part + 1e-12)) ** (1.0 / self.alpha)
        return gain