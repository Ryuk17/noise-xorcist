'''
Author: Ryuk
Date: 2026-02-17 15:03:25
LastEditors: Ryuk
LastEditTime: 2026-02-17 15:16:15
Description: First create
'''

import numpy as np

from ..base import BaseNoiseEstimator



from scipy.interpolate import interp1d

class MSNoiseEstimator(BaseNoiseEstimator):
    """
    Martin, R. (2001). Noise power spectral density estimation based on optimal
	smoothing and minimum statistics. IEEE Transactions on Speech and Audio 
	Processing, 9(5), 504-512
    """

    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1
        
        # 1. 初始化常量参数
        self.D = 150
        self.V = 15
        self.Um = 10
        self.Av = 2.12
        self.alpha_max = 0.96
        self.alpha_min = 0.3
        self.beta_max = 0.8
        self.alpha_corr_fixed = 0.96
        
        # 插值表（用于计算偏差补偿系数）
        x_val = np.array([1, 2, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120, 140, 160])
        Y_M_val = np.array([0, .26, .48, .58, .61, .668, .705, .762, .8, .841, .865, .89, .9, .91])
        # Y_H_val 在 MATLAB 中虽定义但在此近似公式中未直接使用 gamma 函数，保留逻辑
        
        f_interp = interp1d(x_val, Y_M_val, kind='linear', fill_value="extrapolate")
        self.M_D = f_interp(self.D)
        self.M_V = f_interp(self.V)
        
        # 2. 初始化状态变量
        self.alpha_corr = 0.96
        self.alpha = self.alpha_corr_fixed * np.ones(self.fft_bins)
        self.P = np.zeros(self.fft_bins)
        self.Pbar = np.zeros(self.fft_bins)
        self.Psqbar = np.zeros(self.fft_bins)
        self.actmin = np.zeros(self.fft_bins)
        self.actmin_sub = np.zeros(self.fft_bins)
        self.Pmin_u = np.zeros(self.fft_bins)
        self.noise_ps = np.zeros(self.fft_bins)
        
        self.subwc = 1
        self.u = 0  # Python 使用 0-indexed
        self.lmin_flag = np.zeros(self.fft_bins, dtype=bool)
        self.minact = np.zeros((self.fft_bins, self.Um))
        
        self.eps = 1e-12
        self._first_frame = True

    def estimate_noise(self, ns_ps):
        """
        ns_ps: 当前帧带噪信号功率谱
        """
        if self._first_frame:
            self.P = ns_ps.copy()
            self.Pbar = ns_ps.copy()
            self.Psqbar = ns_ps.copy() ** 2
            self.actmin = ns_ps.copy()
            self.actmin_sub = ns_ps.copy()
            self.Pmin_u = ns_ps.copy()
            self.noise_ps = ns_ps.copy()
            self.minact[:] = np.tile(ns_ps[:, np.newaxis], (1, self.Um))
            self._first_frame = False
            return self.noise_ps

        # --- 1. 计算最优平滑因子的修正系数 ---
        alpha_corr_t = 1.0 / (1.0 + (np.sum(self.P) / (np.sum(ns_ps) + self.eps) - 1.0)**2)
        self.alpha_corr = 0.7 * self.alpha_corr + 0.3 * max(alpha_corr_t, 0.7)

        # --- 2. 计算最优自适应平滑因子 ---
        # Eq. 12: 基于 SNR 的平滑系数
        snr_est = self.P / (self.noise_ps + self.eps)
        self.alpha = (self.alpha_max * self.alpha_corr) / ((snr_est - 1.0)**2 + 1.0)
        self.alpha = np.maximum(self.alpha_min, self.alpha)

        # --- 3. 功率谱平滑 ---
        self.P = self.alpha * self.P + (1.0 - self.alpha) * ns_ps

        # --- 4. 计算方差与等效自由度 Qeq ---
        beta = np.minimum(self.alpha**2, self.beta_max)
        self.Pbar = beta * self.Pbar + (1.0 - beta) * self.P
        self.Psqbar = beta * self.Psqbar + (1.0 - beta) * (self.P**2)
        
        var_P = np.abs(self.Psqbar - self.Pbar**2)
        Qeqinv = var_P / (2.0 * self.noise_ps**2 + self.eps)
        Qeqinv = np.minimum(Qeqinv, 0.5)
        Qeq = 1.0 / (Qeqinv + self.eps)

        # --- 5. 计算偏差补偿因子 Bmin 和 Bc ---
        Qeq_tild = (Qeq - 2.0 * self.M_D) / (1.0 - self.M_D + self.eps)
        Qeq_tild_sub = (Qeq - 2.0 * self.M_V) / (1.0 - self.M_V + self.eps)
        
        Bmin = 1.0 + (self.D - 1.0) * 2.0 / (Qeq_tild + self.eps)
        Bmin_sub = 1.0 + (self.V - 1.0) * 2.0 / (Qeq_tild_sub + self.eps)
        
        Qinv_bar = np.mean(1.0 / (Qeq + self.eps))
        Bc = 1.0 + self.Av * np.sqrt(Qinv_bar)

        # --- 6. 最小值追踪逻辑 ---
        # 判断当前是否找到新的局部最小值
        k_mod = (self.P * Bmin * Bc) < self.actmin
        
        self.actmin_sub[k_mod] = self.P[k_mod] * Bmin_sub[k_mod] * Bc
        self.actmin[k_mod] = self.P[k_mod] * Bmin[k_mod] * Bc

        if self.subwc == self.V:
            # 一个子窗口结束
            self.lmin_flag[k_mod] = False
            self.minact[:, self.u] = self.actmin
            self.Pmin_u = np.min(self.minact, axis=1)
            
            # 计算噪声斜率最大值 (Eq. 20 附近的逻辑)
            if Qinv_bar < 0.03: noise_slope_max = 8.0
            elif Qinv_bar < 0.05: noise_slope_max = 4.0
            elif Qinv_bar < 0.06: noise_slope_max = 2.0
            else: noise_slope_max = 1.2
            
            # 更新 Pmin_u
            test = self.lmin_flag & (self.actmin_sub < (noise_slope_max * self.Pmin_u)) & (self.actmin_sub > self.Pmin_u)
            self.Pmin_u[test] = self.actmin_sub[test]
            
            for i in range(self.Um):
                self.minact[test, i] = self.actmin_sub[test]
            
            self.actmin[test] = self.actmin_sub[test]
            
            self.lmin_flag[:] = False
            self.subwc = 1
            self.actmin = self.P.copy()
            self.actmin_sub = self.P.copy()
            self.u = (self.u + 1) % self.Um
        else:
            if self.subwc > 1:
                self.lmin_flag[k_mod] = True
                self.noise_ps = np.minimum(self.actmin_sub, self.Pmin_u)
                self.Pmin_u = self.noise_ps.copy()
            self.subwc += 1

        return self.noise_ps.copy()