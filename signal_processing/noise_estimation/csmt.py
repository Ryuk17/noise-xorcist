'''
Author: Ryuk
Date: 2026-02-17 15:23:37
LastEditors: Ryuk
LastEditTime: 2026-02-17 15:45:10
Description: First create
'''


import numpy as np
from ..base import BaseNoiseEstimator


class CSMTNoiseEstimator(BaseNoiseEstimator):
    """
    Doblinger, G. (1995). Computationally efficient speech enhancement by 
	spectral minima tracking in subbands. Proc. Eurospeech, 2, 1513-1516.
    """

    def __init__(self, n_fft, alpha=0.7, beta=0.96, gamma=0.998):
        """
        参数:
            self.fft_bins (int): 频点数
            alpha (float): 功率谱平滑系数 (取值 0~1)
            beta (float): 修正因子/下限参数
            gamma (float): 最小值上升的时间常数 (接近 1)
        """
        super().__init__()
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # 状态变量初始化
        self.pxk_old = np.zeros(self.fft_bins)
        self.pnk_old = np.zeros(self.fft_bins)
        self.noise_ps = np.zeros(self.fft_bins)
        
        self._is_initialized = False

    def estimate_noise(self, ns_ps):
        """
        ns_ps: 当前帧的功率谱
        """
        # 第一帧初始化
        if not self._is_initialized:
            self.pxk_old = ns_ps.copy()
            self.pnk_old = ns_ps.copy()
            self.noise_ps = ns_ps.copy()
            self._is_initialized = True
            return self.noise_ps

        # 1. 功率谱平滑
        pxk = self.alpha * self.pxk_old + (1 - self.alpha) * ns_ps
        
        # 2. 连续最小值跟踪逻辑 (向量化实现)
        # 如果旧的噪声估计值 <= 当前平滑功率，则缓慢上升 (寻找可能的后续最小值)
        # 如果旧的噪声估计值 > 当前平滑功率，则立即跟进 (发现新的更小的点)
        
        # 计算上升路径的值
        pnk_up = (self.gamma * self.pnk_old) + \
                 ((1 - self.gamma) / (1 - self.beta)) * (pxk - self.beta * self.pxk_old)
        
        # 使用 np.where 替代 MATLAB 的 for 循环
        pnk = np.where(self.pnk_old <= pxk, pnk_up, pxk)
        
        # 3. 更新状态
        self.pxk_old = pxk.copy()
        self.pnk_old = pnk.copy()
        self.noise_ps = pnk.copy()

        return self.noise_ps.copy()