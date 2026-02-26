'''
Author: Ryuk
Date: 2026-02-18 12:53:55
LastEditors: Ryuk
LastEditTime: 2026-02-26 20:56:03
Description: First create
'''

from ..base import BaseSpectralGainEstimator

import numpy as np
import scipy.io.wavfile as wav
from scipy.special import gamma, i0, hyp1f1

class STSAWeuclidSpectralGainEstimator(BaseSpectralGainEstimator):
    """
        Loizou, P. (2005). Speech enhancement based on perceptually motivated 
        Bayesian estimators of the speech magnitude spectrum. IEEE Trans. on Speech 
        and Audio Processing, 13(5), 857-869.
    """
    def __init__(self, n_fft, p=-1.0, aa=0.98, mu=0.98, eta=0.15, eps=1e-12):
        super().__init__()
        if p <= -2:
            raise ValueError("p must be greater than -2")
            
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1

        self.p = p
        self.aa = aa
        self.mu = mu
        self.eta = eta
        self.ksi_min = 10**(-25/10) # 先验信噪比下限
        self.ksi_max = 10**(20/10)  # a priori SNR 的上限
        
        # 预计算常数 CC
        # CC = gamma((p+3)/2) / gamma(p/2+1)
        self.CC = gamma((p + 3) / 2) / gamma(p / 2 + 1)
        
        self.xk_prev = None
        self.is_first_frame = True

        self.eps = eps

    def compute_gain(self, frame_psd, noise_psd):
        # 1. 验后信噪比
        gammak = np.minimum(frame_psd / (noise_psd + self.eps), 40.0)
        
        # 2. 验前信噪比 (Decision-Directed)
        if self.is_first_frame:
            ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1, 0)
            self.is_first_frame = False
        else:
            ksi = self.aa * self.xk_prev / (noise_psd + self.eps) + \
                  (1 - self.aa) * np.maximum(gammak - 1, 0)
        ksi = np.maximum(self.ksi_min, ksi)
        ksi = np.minimum(self.ksi_max, ksi)
            
        # 3. VAD
        log_sigma_k = gammak * ksi / (1 + ksi + self.eps) - np.log(1 + ksi + self.eps)
        vad_decision = np.sum(log_sigma_k) / (self.n_fft // 2) # 对应 MATLAB 的 len
        
        # 4. 计算增益 hw
        vk = ksi * gammak / (1 + ksi + self.eps)
        
        if self.p == -1:
            # 当 p = -1 时的快速简化公式
            # hw = CC * sqrt(vk) / (gammak * exp(-vk/2) * I0(vk/2))
            # 注意：MATLAB 代码中用了 besseli(0, vk/2)
            # 为了稳定，我们使用带有 exp 补偿的 i0e
            import scipy.special as sp
            denom = gammak * sp.i0e(vk / 2.0)
            hw = self.CC * np.sqrt(vk + self.eps) / (denom + self.eps)
        else:
            # 通用情况使用合流超几何函数 hyp1f1(a, b, x)
            # numer = CC * sqrt(vk) * hyp1f1(-(p+1)/2, 1, -vk)
            # denom = gammak * hyp1f1(-p/2, 1, -vk)
            numer = self.CC * np.sqrt(vk + self.eps) * hyp1f1(-(self.p + 1) / 2, 1, -vk)
            denom = gammak * hyp1f1(-self.p / 2, 1, -vk)
            hw = numer / (denom + self.eps)
            
        # 增益限制
        hw = np.nan_to_num(hw, nan=0.0)
        gain = np.clip(hw, 0.0, 1.0)
        
        # 状态更新
        self.xk_prev = (gain**2) * frame_psd
        
        return hw, vad_decision