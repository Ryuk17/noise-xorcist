'''
Author: Ryuk
Date: 2026-02-18 12:55:02
LastEditors: Ryuk
LastEditTime: 2026-02-18 13:48:46
Description: First create
'''
from base import BaseSpectralGainEstimator

import numpy as np
from scipy.special import gamma, hyp1f1, exp1
from scipy.optimize import brentq

class STSAWlrSpectralGainEstimator(BaseSpectralGainEstimator):
    """
        Loizou, P. (2005). Speech enhancement based on perceptually motivated 
        Bayesian estimators of the speech magnitude spectrum. IEEE Trans. on Speech 
        and Audio Processing, 13(5), 857-869.
    """

    def __init__(self, aa=0.98, ksi_min_db=-25):
        """
        参数:
            aa (float): 决策定向 (Decision-Directed) 算法的平滑因子。
            ksi_min_db (float): 先验信噪比的下限 (dB)。
        """
        super().__init__()
        self.aa = aa
        self.ksi_min = 10**(ksi_min_db / 10)
        self.xk_prev = None
        self.gamma_15 = gamma(1.5)

    def _solve_single_bin(self, e_log_x, e_x):
        """
        针对单个频点解方程: f(x) = log(x) + a - b/x = 0
        """
        # 定义目标函数
        def objective(x):
            return np.log(x + 1e-12) + e_log_x - (e_x / (x + 1e-12))
        
        try:
            # 使用 Brent 寻根法。区间选择：[极小值, 极大值]
            # 这里极大值取 e_x * 10 作为一个合理的搜索边界
            return brentq(objective, 1e-8, max(e_x * 10, 1.0))
        except (ValueError, RuntimeError):
            # 如果寻根失败，返回 0.001 作为频谱底噪（Spectral Floor）
            return 0.001

    def compute_gain(self, frame_psd, noise_psd):
        """
        计算 WLR 谱增益
        """
        sig = np.sqrt(frame_psd)
        gammak = np.minimum(frame_psd / (noise_psd + 1e-12), 40.0)
        
        # 1. 估计验前信噪比 ksi
        if self.xk_prev is None:
            ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1, 0)
        else:
            ksi = self.aa * (self.xk_prev / (noise_psd + 1e-12)) + \
                  (1 - self.aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(self.ksi_min, ksi)
            
        vk = (ksi / (1 + ksi + 1e-12)) * gammak
        
        # 2. 计算期望值 E[X] 和 E[log X]
        # lk05 对应 MATLAB 代码中的 sqrt(vk)*Yk/gammak
        lk05 = np.sqrt(vk) * sig / (gammak + 1e-12)
        
        # E[x] = gamma(1.5) * lk05 * 1F1(-0.5, 1, -vk)
        ex = self.gamma_15 * lk05 * hyp1f1(-0.5, 1, -vk)
        
        # E[log x] = 1 - 0.5 * (2*log(lk05) + log(vk) + expint(vk))
        e_log_x = 1.0 - 0.5 * (2.0 * np.log(lk05 + 1e-12) + np.log(vk + 1e-12) + exp1(np.maximum(vk, 1e-10)))
        
        # 3. 对每个频点进行寻根求解 (WLR 核心步骤)
        # 只计算前半部分频谱以加速
        num_bins = len(frame_psd)
        half_len = num_bins // 2 + 1
        x_hat = np.zeros(num_bins)
        
        for i in range(half_len):
            x_hat[i] = self._solve_single_bin(e_log_x[i], ex[i])
            
        # 镜像对称
        if num_bins % 2 == 0:
            x_hat[half_len:] = np.flip(x_hat[1:half_len-1])
        else:
            x_hat[half_len:] = np.flip(x_hat[1:half_len])

        # 4. 转换回增益 Gain = 估计幅度 / 带噪幅度
        gain = x_hat / (sig + 1e-12)
        gain = np.clip(gain, 0.0, 1.0)
        
        # 更新状态
        self.xk_prev = (gain**2) * frame_psd
        
        return gain