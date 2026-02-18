'''
Author: Ryuk
Date: 2026-02-18 12:55:42
LastEditors: Ryuk
LastEditTime: 2026-02-18 14:10:11
Description: First create
'''

import numpy as np
from scipy.special import exp1
from base import BaseSpectralGainEstimator


class OMLSASpectralGainEstimator(BaseSpectralGainEstimator):
    """
    Cohen I, Berdugo B. Speech enhancement for non-stationary noise environments[J]. 
    Signal processing, 2001, 81(11): 2403-2418.
    """

    def __init__(self, fs=16000, n_fft=512, alpha_eta=0.95, alpha_xi=0.7, eta_min_db=-18):
        """
        参数:
            fs (int): 采样率
            n_fft (int): FFT点数
            alpha_eta (float): Decision-Directed 先验信噪比平滑因子
            alpha_xi (float): 长期先验信噪比平滑因子 (用于计算SPP)
            eta_min_db (float): 先验信噪比下限
        """
        super().__init__()
        self.n_fft = n_fft
        self.M21 = n_fft // 2 + 1
        
        # 参数初始化
        self.alpha_eta = alpha_eta
        self.alpha_xi = alpha_xi
        self.eta_min = 10**(eta_min_db / 10)
        self.gain_floor = np.sqrt(self.eta_min) # GH0 的基础值
        
        # 状态变量
        self.eta_2term = np.ones(self.M21)
        self.xi = np.zeros(self.M21)
        
        # SPP 相关阈值 (根据 Cohen 2003 论文标准设置)
        self.q_max = 0.998
        self.P_min = 0.005

    def compute_gain(self, frame_psd, noise_psd):
        """
        计算 OM-LSA 谱增益
        """
        # 1. 基本信噪比计算
        # 验后信噪比 gamma = |Y|^2 / lambda_d
        gamma = frame_psd / (noise_psd + 1e-10)
        gamma = np.minimum(gamma, 40.0) # 限制上限防止溢出
        
        # 2. 先验信噪比估计 (Decision-Directed)
        # eta = alpha * (G_prev^2 * gamma_prev) + (1-alpha) * max(gamma-1, 0)
        eta = self.alpha_eta * self.eta_2term + (1 - self.alpha_eta) * np.maximum(gamma - 1, 0)
        eta = np.maximum(eta, self.eta_min)
        
        # 计算辅助变量 v
        v = gamma * eta / (1 + eta + 1e-10)
        
        # 3. 计算语音存在概率 (Speech Presence Probability, PH1)
        # 长期先验信噪比平滑
        self.xi = self.alpha_xi * self.xi + (1 - self.alpha_xi) * eta
        
        # 简化版 SPP 计算 (基于原代码逻辑)
        # 实际 IMCRA 会更复杂，这里提取 OMLSA 核心：根据先验分布决定语音是否存在
        # q 是语音缺失概率
        q = np.ones(self.M21) * 0.5 # 默认假设 0.5
        # 根据长期先验信噪比调节 q (简单示例逻辑)
        q[self.xi > 1] = 0.1  # 如果长期信噪比高，语音缺失概率低
        q[self.xi < 0.1] = 0.9 # 反之，语音缺失概率高
        q = np.minimum(q, self.q_max)

        # PH1 = 1 / (1 + (q/(1-q))*(1+eta)*exp(-v))
        ph1 = np.zeros(self.M21)
        idx = q < 0.95
        ph1[idx] = 1.0 / (1.0 + (q[idx] / (1.0 - q[idx] + 1e-10)) * (1.0 + eta[idx]) * np.exp(-v[idx]))
        
        # 4. 计算 LSA 增益 (GH1)
        gh1 = np.ones(self.M21)
        # 对于 v > 0, G = (eta/(1+eta)) * exp(0.5 * expint(v))
        idx_positive = v > 0
        gh1[idx_positive] = (eta[idx_positive] / (1 + eta[idx_positive])) * np.exp(0.5 * exp1(v[idx_positive]))
        
        # 5. 综合最终增益 (OM-LSA 公式)
        # G = (GH1 ^ PH1) * (GH0 ^ (1 - PH1))
        # GH0 是语音不存在时的增益补偿 (Gain Floor)
        gh0 = self.gain_floor
        gain = (gh1 ** ph1) * (gh0 ** (1 - ph1))
        
        # 6. 更新状态用于下一帧
        self.eta_2term = (gh1 ** 2) * gamma
        
        return gain