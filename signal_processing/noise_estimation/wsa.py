'''
Author: Ryuk
Date: 2026-02-17 15:24:00
LastEditors: Ryuk
LastEditTime: 2026-02-17 15:47:22
Description: First create
'''

import numpy as np

from base import BaseNoiseEstimator

class WSANoiseEstimator(BaseNoiseEstimator):
    """
    Hirsch, H. and Ehrlicher, C. (1995). Noise estimation techniques for robust 
	speech recognition. Proc. IEEE Int. Conf. Acoust. , Speech, Signal 
	Processing, 153-156.
    """

    def __init__(self, n_fft, alpha_s=0.85, beta=1.5, omin=1.5):
        super().__init__()
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1
        
        self.as_ = alpha_s
        self.beta = beta
        self.omin = omin

        # 状态变量
        self.P = np.zeros(self.fft_bins)        # 平滑后的当前信号功率谱
        self.noise_ps = np.zeros(self.fft_bins) # 估计的噪声功率谱
        
        self._is_initialized = False

    def estimate_noise(self, ns_ps):
        """
        ns_ps: 当前帧的功率谱
        """
        # 第一帧初始化逻辑
        if not self._is_initialized:
            self.P = ns_ps.copy()
            self.noise_ps = ns_ps.copy()
            self._is_initialized = True
            return self.noise_ps

        # 1. 对输入功率谱进行递归平滑
        # P[k] = as * P_old[k] + (1 - as) * ns_ps[k]
        self.P = self.as_ * self.P + (1 - self.as_) * ns_ps

        # 2. 条件更新噪声估计
        # 判定准则：如果当前平滑功率 P 小于 噪声估计的 beta 倍，则认为是噪声区域
        update_mask = self.P < (self.beta * self.noise_ps)
        
        # 仅对满足条件的频点更新噪声
        # noise[index] = as * noise_old[index] + (1 - as) * P[index]
        self.noise_ps[update_mask] = (self.as_ * self.noise_ps[update_mask] + 
                                     (1 - self.as_) * self.P[update_mask])

        return self.noise_ps.copy()