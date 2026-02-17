'''
Author: Ryuk
Date: 2026-02-17 15:21:11
LastEditors: Ryuk
LastEditTime: 2026-02-17 15:39:17
Description: First create
'''


import numpy as np
from scipy.special import exp1 # 对应 MATLAB 的 expint

import numpy as np

from base import BaseNoiseEstimator

class IMCRA(BaseNoiseEstimator):
    """
    Cohen, I. (2003). Noise spectrum estimation in adverse environments: 
	Improved minima controlled recursive averaging. IEEE Transactions on Speech 
	and Audio Processing, 11(5), 466-475.
    """

    def __init__(self, n_fft, alpha_d=0.85, alpha_s=0.9, U=8, V=15, 
                 Bmin=1.66, gamma0=4.6, gamma1=3, psi0=1.67, 
                 alpha=0.92, beta=1.47):
        super().__init__()
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1

        # 算法超参数
        self.alpha_d = alpha_d
        self.alpha_s = alpha_s
        self.U = U
        self.V = V
        self.Bmin = Bmin
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.psi0 = psi0
        self.alpha = alpha
        self.beta = beta
        
        # 频率平滑窗 (3点 Hanning)
        b = np.array([0.25, 0.5, 0.25])
        self.b = b / np.sum(b)
        
        # 状态变量初始化
        self.n = 2
        self.j = 0
        self.u1 = 0
        self.u2 = 0
        
        self.noise_ps = None
        self.noise_tild = None
        self.S = None
        self.S_tild = None
        self.Smin = None
        self.Smin_tild = None
        self.Smin_sw = None
        self.Smin_sw_tild = None
        self.GH1 = np.ones(self.fft_bins)
        self.gamma_old = np.ones(self.fft_bins)
        
        # 最小值追踪存储
        self.stored_min = None
        self.stored_min_tild = None
        
        self._is_initialized = False

    def _frequency_smoothing(self, ps):
        """对频率轴进行 3 点平滑"""
        res = ps.copy()
        # 内部频点平滑
        res[1:-1] = (self.b[0] * ps[:-2] + 
                     self.b[1] * ps[1:-1] + 
                     self.b[2] * ps[2:])
        return res

    def estimate_noise(self, ns_ps):
        if not self._is_initialized:
            self.noise_ps = ns_ps.copy()
            self.noise_tild = ns_ps.copy()
            Sf = self._frequency_smoothing(ns_ps)
            self.S = Sf.copy()
            self.S_tild = Sf.copy()
            self.Smin = Sf.copy()
            self.Smin_tild = Sf.copy()
            self.Smin_sw = Sf.copy()
            self.Smin_sw_tild = Sf.copy()
            
            # 初始化存储矩阵
            max_val = np.max(ns_ps)
            self.stored_min = np.ones((self.fft_bins, self.U)) * max_val
            self.stored_min_tild = np.ones((self.fft_bins, self.U)) * max_val
            
            self._is_initialized = True
            return self.noise_ps.copy()

        # 1. 计算先验和后验 SNR
        gamma_post = ns_ps / (self.noise_ps + 1e-12)
        # Eq 32: 计算先验 SNR (Decision-Directed 估计)
        eps_cap = self.alpha * (self.GH1**2) * self.gamma_old + (1 - self.alpha) * np.maximum(gamma_post - 1, 0)
        self.gamma_old = gamma_post.copy()
        
        # 2. 计算增益函数 GH1
        v = gamma_post * eps_cap / (1 + eps_cap + 1e-12)
        exp_int = exp1(v + 1e-12)
        self.GH1 = eps_cap * np.exp(0.5 * exp_int) / (1 + eps_cap + 1e-12)
        
        # 3. 第一次迭代：粗略语音活跃检测
        Sf = self._frequency_smoothing(ns_ps)
        self.S = self.alpha_s * self.S + (1 - self.alpha_s) * Sf
        self.Smin = np.minimum(self.Smin, self.S)
        self.Smin_sw = np.minimum(self.Smin_sw, self.S)
        
        gamma_min = ns_ps / (self.Bmin * self.Smin + 1e-12)
        psi = self.S / (self.Bmin * self.Smin + 1e-12)
        
        # 指示函数 I (基于双阈值判定语音是否存在)
        I = ((gamma_min < self.gamma0) & (psi < self.psi0)).astype(float)
        
        # 4. 第二次迭代：精细估计
        # 计算受限平滑 Sf_tild
        conv_I_ps = self._frequency_smoothing(I * ns_ps)
        conv_I = self._frequency_smoothing(I)
        # 避免除以 0
        Sf_tild = np.where(conv_I > 0, conv_I_ps / (conv_I + 1e-12), self.S_tild)
        
        self.S_tild = self.alpha_s * self.S_tild + (1 - self.alpha_s) * Sf_tild
        self.Smin_tild = np.minimum(self.Smin_tild, self.S_tild)
        self.Smin_sw_tild = np.minimum(self.Smin_sw_tild, self.S_tild)
        
        # 5. 计算语音缺失先验概率 q
        gamma_min_tild = ns_ps / (self.Bmin * self.Smin_tild + 1e-12)
        psi_tild = self.S / (self.Bmin * self.Smin_tild + 1e-12)
        
        q = np.zeros(self.fft_bins)
        # 情况 1: 判定为无语音
        q[gamma_min_tild <= 1] = 1
        # 情况 2: 转换区域
        mask_trans = (gamma_min_tild > 1) & (gamma_min_tild < self.gamma1) & (psi_tild < self.psi0)
        q[mask_trans] = (self.gamma1 - gamma_min_tild[mask_trans]) / (self.gamma1 - 1)
        
        # 6. 计算条件语音存在概率 p
        # 基于全概率公式和高斯模型推导
        p = np.zeros(self.fft_bins)
        non_abs_mask = (q < 1) # 只有 q < 1 时才计算，避免除以 0
        if np.any(non_abs_mask):
            term1 = q[non_abs_mask] / (1 - q[non_abs_mask] + 1e-12)
            term2 = 1 + eps_cap[non_abs_mask]
            term3 = np.exp(-v[non_abs_mask])
            p[non_abs_mask] = 1.0 / (1.0 + term1 * term2 * term3)

        # 7. 更新噪声估计
        alpha_d_curr = self.alpha_d + (1 - self.alpha_d) * p
        self.noise_tild = alpha_d_curr * self.noise_tild + (1 - alpha_d_curr) * ns_ps
        # 引入修正系数 beta
        self.noise_ps = self.beta * self.noise_tild
        
        # 8. 窗口滑动逻辑 (每过 V 帧更新一次 U 个小窗口的最小值)
        self.j += 1
        if self.j >= self.V:
            # 更新第一个迭代的最小值
            self.stored_min[:, self.u1] = self.Smin_sw
            self.u1 = (self.u1 + 1) % self.U
            self.Smin = np.min(self.stored_min, axis=1)
            self.Smin_sw = self.S.copy()
            
            # 更新第二个迭代的最小值
            self.stored_min_tild[:, self.u2] = self.Smin_sw_tild
            self.u2 = (self.u2 + 1) % self.U
            self.Smin_tild = np.min(self.stored_min_tild, axis=1)
            self.Smin_sw_tild = self.S_tild.copy()
            
            self.j = 0

        return self.noise_ps.copy()