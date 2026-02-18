'''
Author: Ryuk
Date: 2026-02-18 12:54:43
LastEditors: Ryuk
LastEditTime: 2026-02-18 13:41:19
Description: First create
'''
from base import BaseSpectralGainEstimator

import numpy as np
from scipy.special import gamma, hyp1f1

class STSAWCoshSpectralGainEstimator(BaseSpectralGainEstimator):
    """
        Loizou, P. (2005). Speech enhancement based on perceptually motivated 
        Bayesian estimators of the speech magnitude spectrum. IEEE Trans. on Speech 
        and Audio Processing, 13(5), 857-869.
    """

    def __init__(self, p=-0.5, aa=0.98, ksi_min_db=-25):
        """
        参数:
            p (float): 幂指数参数，必须大于 -1。常用值如 -0.5。
            aa (float): 决策定向 (Decision-Directed) 算法的平滑因子。
            ksi_min_db (float): 先验信噪比的下限 (dB)，防止增益过小产生音乐噪声。
        """
        super().__init__()
        if p <= -1:
            raise ValueError("参数 p 必须大于 -1")
            
        self.p = p
        self.aa = aa
        self.ksi_min = 10**(ksi_min_db / 10)
        
        # 预计算 Gamma 常数项
        # CC2 = sqrt( gamma((p+3)/2) / gamma((p+1)/2) )
        self.cc2 = np.sqrt(gamma((p + 3) / 2) / gamma((p + 1) / 2))
        
        # 状态变量：保存上一帧增强后的功率谱，用于 DD 估计
        self.xk_prev = None

    def compute_gain(self, frame_psd, noise_psd):
        """
        计算 Weighted-Cosh 谱增益
        
        参数:
            frame_psd (array-like): 当前帧的带噪功率谱 |Y(k)|^2
            noise_psd (array-like): 估计的噪声功率谱 lambda_d(k)
            
        返回:
            gain (array-like): 计算得到的谱增益 G(k)
        """
        # 1. 计算验后信噪比 (Posteriori SNR) γk = |Y|^2 / lambda_d
        # 限制最大值为 40 (约 16dB)，防止 hyp1f1 溢出
        gammak = np.minimum(frame_psd / (noise_psd + 1e-12), 40.0)
        
        # 2. 估计验前信噪比 (Priori SNR) ξk - 决策定向法 (DD)
        if self.xk_prev is None:
            # 第一帧使用直接估计
            ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1, 0)
        else:
            # ksi = aa * (X_prev / Noise) + (1 - aa) * max(gamma - 1, 0)
            ksi = self.aa * (self.xk_prev / (noise_psd + 1e-12)) + \
                  (1 - self.aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(self.ksi_min, ksi)
            
        # 3. 计算中间变量 vk = (ξ / (1 + ξ)) * γ
        vk = (ksi / (1 + ksi + 1e-12)) * gammak
        
        # 4. 计算加权 Cosh 增益函数
        # 使用合流超几何函数 hyp1f1 (1F1)
        hyp_num = hyp1f1(-(self.p + 1) / 2, 1, -vk)
        hyp_den = hyp1f1(-(self.p - 1) / 2, 1, -vk)
        
        # 数值稳定性：确保 hyp1f1 结果非负
        hyp_num = np.maximum(hyp_num, 0)
        hyp_den = np.maximum(hyp_den, 1e-12)
        
        # 增益公式：G = CC2 * sqrt(vk * 1F1_num) / (gammak * sqrt(1F1_den))
        numer = self.cc2 * np.sqrt(vk * hyp_num + 1e-12)
        denom = gammak * np.sqrt(hyp_den)
        
        gain = numer / (denom + 1e-12)
        
        # 限制增益范围在 [0, 1] 之间
        gain = np.nan_to_num(gain, nan=0.0)
        gain = np.clip(gain, 0.0, 1.0)
        
        # 5. 更新状态：保存当前帧增强后的语音功率谱估计
        # Xk_prev = G^2 * |Y|^2
        self.xk_prev = (gain**2) * frame_psd
        
        return gain