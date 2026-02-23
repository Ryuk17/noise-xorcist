'''
Author: Ryuk
Date: 2026-02-18 12:53:03
LastEditors: Ryuk
LastEditTime: 2026-02-23 17:46:10
Description: First create
'''

from ..base import BaseSpectralGainEstimator

import numpy as np
from scipy.special import exp1

class LogMMSESpectralEstimator(BaseSpectralGainEstimator):
    """
        Ephraim, Y. and Malah, D. (1985). Speech enhancement using a minimum 
        mean-square error log-spectral amplitude estimator. IEEE Trans. Acoust., 
        Speech, Signal Process., ASSP-23(2), 443-445.
    """
    def __init__(self, n_fft, aa=0.98, mu=0.98, eta=0.15, eps=1e-12):
        super().__init__()
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1

        self.aa = aa                # Decision-Directed 因子
        self.mu = mu                # 噪声平滑因子
        self.eta = eta              # VAD 阈值
        self.ksi_min = 10**(-25/10) # a priori SNR 的下限
        self.ksi_max = 10**(20/10)  # a priori SNR 的上限
        
        self.xk_prev = None         # 上一帧纯净信号功率
        self.is_first_frame = True

        self.eps = eps

    def compute_gain(self, frame_psd, noise_psd):
        """
        计算 Log-MMSE 增益
        """
        # 1. 计算验后信噪比 (Posteriori SNR)
        gammak = np.minimum(frame_psd / (noise_psd + self.eps), 40)
        
        # 2. 估计验前信噪比 (Priori SNR) - DD 法
        if self.is_first_frame:
            ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1, 0)
            self.is_first_frame = False
        else:
            # ksi = aa * (X_prev / Noise) + (1-aa) * max(gamma-1, 0)
            ksi = self.aa * (self.xk_prev / (noise_psd + self.eps)) + \
                  (1 - self.aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(self.ksi_min, ksi)
            ksi = np.minimum(self.ksi_max, ksi)
            
        # 3. VAD 逻辑 (基于似然比累加)
        log_sigma_k = gammak * ksi / (1 + ksi + self.eps) - np.log(1 + ksi + self.eps)

        vad_decision = np.sum(log_sigma_k) / (self.fft_bins) 
        
        # 4. 计算 Log-MMSE 增益函数
        # 公式: G = (ksi/(1+ksi)) * exp(0.5 * E1( (ksi/(1+ksi)) * gammak ))
        A = ksi / (1 + ksi + self.eps)
        vk = A * gammak
        
        # 处理 exp1(vk) 的数值问题，当 vk 非常小时 exp1 会趋向无穷
        # 我们给 vk 设置一个极小的下限以保证计算稳定
        ei_vk = 0.5 * exp1(np.maximum(vk, self.eps))
        gain = A * np.exp(ei_vk)
        
        # 限制增益范围
        gain = np.clip(gain, 0.0, 1.0)
        
        # 5. 更新状态
        self.xk_prev = (gain**2) * frame_psd
        
        return gain, vad_decision
