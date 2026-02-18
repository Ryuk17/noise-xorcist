'''
Author: Ryuk
Date: 2026-02-17 15:21:05
LastEditors: Ryuk
LastEditTime: 2026-02-17 15:28:16
Description: First create
'''

import numpy as np

from base import BaseNoiseEstimator

class MCRA2NoiseEstimator(BaseNoiseEstimator):
    """
    Cohen, I. (2002). Noise estimation by minima controlled recursive averaging 
	for robust speech enhancement. IEEE Signal Processing Letters, 9(1), 12-15.
    """

    def __init__(self, n_fft, sample_rate=16000, ad=0.95, as_=0.8, ap=0.2, 
                 beta=0.8, gamma=0.998, alpha=0.7):
        """
        参数:
            self.fft_bins (int): 频点数 (通常为 n_fft // 2 + 1)
            Srate (int): 采样率，用于计算频率分辨率和分段阈值
        """
        super().__init__()
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1
        self.sample_rate = sample_rate

        self.ad = ad
        self.as_ = as_
        self.ap = ap
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha

        # 1. 计算频率相关的阈值 delta
        # 低频 ( < 3kHz) 阈值较低 (2)，高频阈值较高 (5)
        freq_res = self.sample_rate / (2 * (self.fft_bins - 1)) if self.fft_bins > 1 else self.sample_rate
        k_1khz = int(np.floor(1000 / freq_res))
        k_3khz = int(np.floor(3000 / freq_res))
        
        # 构造 delta 向量
        d = np.ones(self.fft_bins) * 5 # 默认设为 5
        d[:k_3khz] = 2           # 3kHz 以下设为 2
        self.delta = d

        # 2. 初始化状态变量
        self.pk = np.zeros(self.fft_bins)
        self.pxk_old = np.zeros(self.fft_bins)
        self.pnk_old = np.zeros(self.fft_bins)
        self.noise_ps = np.zeros(self.fft_bins)
        
        self._is_initialized = False

    def estimate_noise(self, ns_ps):
        """
        使用 MCRA2 逻辑估计噪声功率谱
        """
        if not self._is_initialized:
            self.pxk_old = ns_ps.copy()
            self.pnk_old = ns_ps.copy()
            self.noise_ps = ns_ps.copy()
            self._is_initialized = True
            return self.noise_ps

        # --- 步骤 1: 平滑当前功率谱 ---
        # pxk: 信号功率谱的平滑估计
        pxk = self.alpha * self.pxk_old + (1 - self.alpha) * ns_ps

        # --- 步骤 2: 连续最小值跟踪 (Minima Tracking) ---
        # 如果当前功率大于之前的最小值，则缓慢上升；否则直接跟进
        pnk = pxk.copy()
        mask = self.pnk_old < pxk
        
        # MATLAB 公式: pnk = gamma*pnk_old + (1-gamma)/(1-beta)*(pxk - beta*pxk_old)
        pnk[mask] = (self.gamma * self.pnk_old[mask]) + \
                   ((1 - self.gamma) / (1 - self.beta)) * \
                   (pxk[mask] - self.beta * self.pxk_old[mask])

        # --- 步骤 3: 语音存在概率计算 ---
        # 计算比值 Srk
        Srk = pxk / (pnk + 1e-12)
        
        # 判断指示函数 Ikl (大于阈值判定为语音)
        Ikl = (Srk > self.delta).astype(float)
        
        # 平滑得到语音存在概率 pk
        self.pk = self.ap * self.pk + (1 - self.ap) * Ikl

        # --- 步骤 4: 噪声更新 ---
        # 结合语音概率计算自适应平滑因子 adk
        adk = self.ad + (1 - self.ad) * self.pk
        self.noise_ps = adk * self.noise_ps + (1 - adk) * pxk

        # --- 状态更新 ---
        self.pxk_old = pxk.copy()
        self.pnk_old = pnk.copy()

        return self.noise_ps.copy()