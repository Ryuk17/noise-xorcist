'''
Author: Ryuk
Date: 2026-02-17 15:47:46
LastEditors: Ryuk
LastEditTime: 2026-02-18 12:59:38
Description: First create
'''

import numpy as np
from base import BaseNoiseEstimator

import numpy as np
from scipy.signal.windows import triang

class CFRNoiseEstimator(BaseNoiseEstimator):
    """
    Sorensen, K. and Andersen, S. (2005). Speech enhancement with natural 
	sounding residual noise based on connected time-frequency speech presence 
	regions. EURASIP J. Appl. Signal Process., 18, 2954-2964.
    """

    def __init__(self, n_fft, D=7, U=5, V=8, gamma1=6, gamma2=0.5, 
                 alpha_max=0.96, alpha_min=0.3, beta_min=0.7, alpha_c=0.7):
        super().__init__()
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1

        self.D = D
        self.U = U
        self.V = V
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        
        # 1. 构造谱域平滑三角形窗 (Eq. 4)
        # 对应 triang(2*D+1)
        win = triang(2 * D + 1)
        self.b = win / np.sum(win)
        
        # 2. 初始化状态变量
        self.alpha_c = alpha_c
        self.noise_ps = np.zeros(self.fft_bins)
        self.Rmin_old = 1.0
        self.Pmin = np.zeros(self.fft_bins)
        self.Pmin_sw = np.zeros(self.fft_bins)
        self.SmthdP = np.zeros(self.fft_bins)  # 对应 SmthdP (P)
        
        self.stored_min = np.zeros((self.fft_bins, U))
        self.u1 = 0
        self.j = 0
        
        self.eps = 1e-12
        self._is_initialized = False

    def _spectral_smoothing(self, x):
        """对频率轴进行卷积平滑，对应 MATLAB 的 smoothing 函数"""
        # 使用 numpy.convolve 实现，mode='same' 保证输出长度不变
        return np.convolve(x, self.b, mode='same')

    def estimate_noise(self, ns_ps):
        if not self._is_initialized:
            self.noise_ps = ns_ps.copy()
            self.Pmin = ns_ps.copy()
            self.Pmin_sw = ns_ps.copy()
            self.SmthdP = ns_ps.copy()
            self.stored_min[:] = np.tile(ns_ps[:, np.newaxis], (1, self.U))
            self._is_initialized = True
            return self.noise_ps

        # --- 步骤 1: 谱域平滑 ---
        P_y = self._spectral_smoothing(ns_ps)

        # --- 步骤 2: 计算自适应平滑因子 alpha ---
        # R 为平滑谱与原始谱的能量比
        R = np.sum(self.SmthdP) / (np.sum(ns_ps) + self.eps)
        alpha_c_tild = 1.0 / (1.0 + (R - 1.0)**2)
        self.alpha_c = self.alpha_c * 0.7 + 0.3 * max(alpha_c_tild, 0.7)
        
        # 基于 SNR 的自适应平滑系数
        snr_est = self.SmthdP / (self.noise_ps + self.eps)
        alpha = (self.alpha_max * self.alpha_c) / ((snr_est - 1.0)**2 + 1.0)
        alpha = np.maximum(self.alpha_min, alpha)

        # --- 步骤 3: 时间域平滑与语音判定 ---
        power_min = np.sum(self.Pmin)
        self.SmthdP = alpha * self.SmthdP + (1.0 - alpha) * P_y
        
        # 更新局部最小值
        self.Pmin = np.minimum(self.Pmin, self.SmthdP)
        self.Pmin_sw = np.minimum(self.Pmin_sw, self.SmthdP)
        
        # 判定条件 1: 当前能量显著高于最小能量
        D_1a = (self.SmthdP > self.gamma1 * self.Pmin)
        # 判定条件 2: 当前能量高于谱底偏移量
        D_2a = (self.SmthdP > (self.Pmin + self.gamma2 * power_min / self.fft_bins))
        
        Decision = D_1a & D_2a

        # --- 步骤 4: 偏差因子 Rmin 估计 ---
        power_noise_ps = np.sum(self.noise_ps)
        Rmin_tild = power_noise_ps / (power_min + self.eps)

        if np.sum(Decision) > 0:
            # 如果判定为有语音，维持旧的偏差因子
            Rmin = self.Rmin_old
        else:
            # 如果判定为无语音，更新偏差因子
            Rmin = self.beta_min * self.Rmin_old + (1.0 - self.beta_min) * Rmin_tild
        
        self.Rmin_old = Rmin

        # --- 步骤 5: 更新噪声估计 ---
        # 默认使用原始周期图 ns_ps，但在判定为语音的频点使用 Bias * Pmin 替换
        self.noise_ps = ns_ps.copy()
        self.noise_ps[Decision] = Rmin * self.Pmin[Decision]

        # --- 步骤 6: 最小值追踪 (子窗口滑动) ---
        self.j += 1
        if self.j >= self.V:
            self.stored_min[:, self.u1] = self.Pmin_sw
            self.u1 = (self.u1 + 1) % self.U
            
            # 从存储的所有子窗口中寻找全局最小值
            self.Pmin = np.min(self.stored_min, axis=1)
            self.Pmin_sw = self.SmthdP.copy()
            self.j = 0

        return self.noise_ps.copy()