'''
Author: Ryuk
Date: 2026-02-15 16:33:45
LastEditors: Ryuk
LastEditTime: 2026-02-16 16:50:50
Description: First create
'''

from base import BaseNoiseEstimator

import numpy as np

class MCRANoiseEstimator(BaseNoiseEstimator):
    def __init__(self, n_fft, alpha_s=0.8, alpha_d=0.95, alpha_p=0.2, L=100, delta=5):
        super().__init__()
        self.as_ = alpha_s
        self.ad = alpha_d
        self.ap = alpha_p
        self.L = L
        self.delta = delta
        self.n_fft = n_fft
        
        # 状态变量：根据 FFT 频点数初始化
        num_bins = n_fft // 2 + 1
        self.n = 1
        self.pk = np.zeros(num_bins)
        self.P = np.zeros(num_bins)
        self.Pmin = np.zeros(num_bins)
        self.Ptmp = np.zeros(num_bins)
        self.noise_ps = np.zeros(num_bins)
        self._initialized = False

    def estimate_noise(self, frame_psd):
        """
        参数 frame_psd: 当前帧的功率谱
        """
        if not self._initialized:
            self.P = frame_psd.copy()
            self.Pmin = frame_psd.copy()
            self.Ptmp = frame_psd.copy()
            self.noise_ps = frame_psd.copy()
            self._initialized = True
            return self.noise_ps

        # MCRA 核心递归逻辑
        self.P = self.as_ * self.P + (1 - self.as_) * frame_psd
        if self.n % self.L == 0:
            self.Pmin = np.minimum(self.Ptmp, self.P)
            self.Ptmp = self.P.copy()
        else:
            self.Pmin = np.minimum(self.Pmin, self.P)
            self.Ptmp = np.minimum(self.Ptmp, self.P)

        Srk = self.P / (self.Pmin + 1e-12)
        Ikl = (Srk > self.delta).astype(float)
        self.pk = self.ap * self.pk + (1 - self.ap) * Ikl
        adk = self.ad + (1 - self.ad) * self.pk
        self.noise_ps = adk * self.noise_ps + (1 - adk) * frame_psd
        
        self.n += 1
        return self.noise_ps.copy()