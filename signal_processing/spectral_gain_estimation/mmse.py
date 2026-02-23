'''
Author: Ryuk
Date: 2026-02-18 12:37:20
LastEditors: Ryuk
LastEditTime: 2026-02-23 22:31:58
Description: First create
'''

from ..base import BaseSpectralGainEstimator

import numpy as np
from scipy.special import i0e, i1e

class MMSESpectralEstimator(BaseSpectralGainEstimator):
    """
        Ephraim, Y. and Malah, D. (1985). Speech enhancement using a minimum 
        mean-square error log-spectral amplitude estimator. IEEE Trans. Acoust., 
        Speech, Signal Process., ASSP-23(2), 443-445.
    """
    def __init__(self, n_fft, aa=0.98, mu=0.98, eta=0.15, qk=0.3, eps=1e-12, spu=True):
        super().__init__()
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1

        self.aa = aa          # DD 估计器的平滑因子
        self.mu = mu          # 噪声更新平滑因子
        self.eta = eta        # VAD 阈值
        self.qk = qk          # 语音不存在的先验概率
        self.qkr = (1 - qk) / qk
        self.spu = spu        # 是否启用语音存在不确定性
        self.ksi_min = 10**(-25/10) # 先验信噪比下限
        self.ksi_max = 10**(20/10)  # a priori SNR 的上限
        
        self.c = np.sqrt(np.pi) / 2
        self.xk_prev = None   # 上一帧的纯净语音功率估计
        self.is_first_frame = True

        self.eps = eps

    def compute_gain(self, frame_psd, noise_psd):
        """
        参数:
            frame_fft: 当前帧的 FFT (复数)
            noise_mu2: 噪声功率谱估计
        """
  
        # 1. 计算验后信噪比 (Posteriori SNR)
        # 防止除 0，限制最大值为 40
        gammak = np.minimum(frame_psd / (noise_psd + self.eps), 40)
        
        # 2. 决策定向 (DD) 估计先验信噪比 (Priori SNR)
        if self.is_first_frame:
            ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1, 0)
            self.is_first_frame = False
        else:
            # ksi = aa * (X_prev/Noise) + (1-aa) * max(gamma-1, 0)
            ksi = self.aa * (self.xk_prev / (noise_psd + self.eps)) + \
                  (1 - self.aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(self.ksi_min, ksi)
            ksi = np.minimum(self.ksi_max, ksi)
            
        # 3. VAD 逻辑
        log_sigma_k = gammak * ksi / (1 + ksi + self.eps) - np.log(1 + ksi + self.eps)
        vad_decision = np.sum(log_sigma_k) / self.fft_bins
        
        # 4. 计算 MMSE 增益函数
        # vk = (ksi / (1 + ksi)) * gammak
        vk = (ksi / (1 + ksi + self.eps)) * gammak
        
        # 使用指数缩放的 Bessel 函数：i0e(x) = exp(-x) * i0(x)
        # 原公式：exp(-0.5*vk) * [(1+vk)*i0(vk/2) + vk*i1(vk/2)]
        # 转换后：(1+vk)*i0e(vk/2) + vk*i1e(vk/2) 
        # 这样可以完美避免原 MATLAB 代码中的 Overflow 风险
        if_part = (1 + vk) * i0e(vk / 2.0) + vk * i1e(vk / 2.0)
        hw = (self.c * np.sqrt(vk + self.eps) / gammak) * if_part
        
        # 5. 语音存在不确定性 (SPU)
        if self.spu:
            # Lambda = (1-qk)/qk * exp(vk) / (1+ksi)
            # 为了数值稳定，使用 np.exp 配合 limit
            evk = np.exp(np.minimum(vk, 100)) 
            lambda_k = self.qkr * evk / (1 + ksi + self.eps)
            p_sap = lambda_k / (1 + lambda_k + self.eps)
            gain = hw * p_sap
        else:
            gain = hw
            
        # 限制增益范围
        gain = np.nan_to_num(gain, nan=0.0, posinf=1.0)
        gain = np.clip(gain, 0.0, 1.0)
        
        # 保存纯净信号功率估计用于下一帧
        self.xk_prev = (gain**2) * frame_psd
        
        return gain, vad_decision

