'''
Author: Ryuk
Date: 2026-02-18 12:55:24
LastEditors: Ryuk
LastEditTime: 2026-02-18 13:55:55
Description: First create
'''

from base import BaseSpectralGainEstimator

import numpy as np
from scipy.special import gamma, factorial

class STSAMisSpectralGainEstimator(BaseSpectralGainEstimator):
    """
        Loizou, P. (2005). Speech enhancement based on perceptually motivated 
        Bayesian estimators of the speech magnitude spectrum. IEEE Trans. on Speech 
        and Audio Processing, 13(5), 857-869.
    """

    def __init__(self, aa=0.98, ksi_min_db=-25, n_terms=40):
        """
        参数:
            aa (float): 决策定向平滑因子。
            ksi_min_db (float): 先验 SNR 下限。
            n_terms (int): 无穷级数展开的项数，默认为 40。
        """
        super().__init__()
        self.aa = aa
        self.ksi_min = 10**(ksi_min_db / 10)
        self.n_terms = n_terms
        self.xk_prev = None

    def _hyperg_term(self, m, val, c_val):
        """
        MATLAB 原代码中的 hyperg(-m, -m, c, val)
        当 a, b 为负整数时，合流超几何函数退化为多项式。
        """
        # _hyperg(-m, -m, c, z) = sum_{n=0}^m ((-m)_n * (-m)_n / (c)_n) * (z^n / n!)
        # 其中 (x)_n 是 Pochhammer 符号
        res = np.ones_like(val)
        poch_m = 1.0
        poch_c = 1.0
        fact_n = 1.0
        z_pow = 1.0
        
        for n in range(1, m + 1):
            poch_m *= (-m + n - 1)
            poch_c *= (c_val + n - 1)
            fact_n *= n
            z_pow *= val
            res += (poch_m * poch_m / (poch_c + 1e-12)) * (z_pow / fact_n)
        return res

    def compute_gain(self, frame_psd, noise_psd):
        """
        计算 MIS 谱增益
        """
        sig = np.sqrt(frame_psd)
        gammak = np.minimum(frame_psd / (noise_psd + 1e-12), 40.0)
        
        # 1. 估计先验 SNR (ksi)
        if self.xk_prev is None:
            ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1, 0)
        else:
            ksi = self.aa * (self.xk_prev / (noise_psd + 1e-12)) + \
                  (1 - self.aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(self.ksi_min, ksi)
            
        vk = (ksi / (1 + ksi + 1e-12)) * gammak
        
        # 2. 准备级数计算所需的变量
        half_len = len(frame_psd) // 2 + 1
        vk_h = vk[:half_len]
        sig_h = sig[:half_len]
        gam_h = gammak[:half_len]
        
        # arg = Y^2 / (4 * gamma^2)
        arg = (sig_h**2) / (4 * gam_h**2 + 1e-12)
        
        sum_j1 = np.zeros(half_len)
        sum_j2 = np.zeros(half_len)
        
        # 3. 计算无穷级数 (Eq. 43)
        for m in range(self.n_terms):
            fact_m = factorial(m)
            vk_pow_m = vk_h**m
            
            # 计算两个超几何多项式项
            d2 = self._hyperg_term(m, arg, 0.5)
            d2_b = self._hyperg_term(m, arg, 1.5)
            
            # J1 累加
            sum_j1 += (vk_pow_m * d2) / fact_m
            
            # J2 累加 (包含 gamma(m+1.5) / gamma(m+1))
            g_term = gamma(m + 1.5) / (fact_m + 1e-12)
            sum_j2 += g_term * vk_pow_m * d2_b / (fact_m + 1e-12)

        # 4. 组合 J1 和 J2
        ev = np.exp(-vk_h)
        j1 = sum_j1 * ev
        j2 = sum_j2 * ev * np.sqrt(vk_h) * sig_h / (gam_h + 1e-12)
        
        # x_hat = exp(J1 + J2) -- 注意：原 MATLAB 代码这里写的是 sig_hat = log(...) 
        # 但在主循环里又把 sig_hat 直接当幅度用了，这里遵循数学逻辑返回幅度估计值
        x_hat_h = np.maximum(np.real(j1 + j2), 1e-5)
        
        # 对称填充
        x_hat = np.zeros(len(frame_psd))
        x_hat[:half_len] = x_hat_h
        x_hat[half_len:] = np.flip(x_hat_h[1 : len(frame_psd) - half_len + 1])
        
        # 计算增益
        gain = x_hat / (sig + 1e-12)
        gain = np.clip(gain, 0.0, 1.0)
        
        # 更新状态
        self.xk_prev = (gain**2) * frame_psd
        
        return gain