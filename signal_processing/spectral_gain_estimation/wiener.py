'''
Author: Ryuk
Date: 2026-02-17 19:15:53
LastEditors: Ryuk
LastEditTime: 2026-02-26 21:46:30
Description: First create
'''


from ..base import BaseSpectralGainEstimator

import numpy as np
import scipy.io.wavfile as wav


class WienerSpectralGainEstimator(BaseSpectralGainEstimator):
    """
        Scalart, P. and Filho, J. (1996). Speech enhancement based on a priori 
        signal to noise estimation. Proc. IEEE Int. Conf. Acoust. Speech, Signal 
        Processing, 629-632.
    """
    def __init__(self, n_fft, a_dd=0.98, mu=0.98, eta=0.15, eps=1e-12):
        super().__init__()
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1
        
        self.a_dd = a_dd            # 验前信噪比平滑因子
        self.mu = mu                # 噪声更新平滑因子
        self.eta = eta              # VAD 阈值
        
        # 状态变量：保存上一帧的信息
        self.G_prev = np.ones(self.fft_bins)
        self.posteri_prev = np.ones(self.fft_bins)
        self.is_first_frame = True

        self.eps = eps

    def compute_gain(self, frame_psd, noise_psd):
        # 2. 计算验后信噪比 (Posteriori SNR)
        # γ(k) = |Y(k)|^2 / E[|N(k)|^2]
        posteri = np.minimum(frame_psd / (noise_psd + self.eps), 40.0)
        posteri_prime = np.maximum(posteri - 1, 0)
        
        # 3. 决策定向 (Decision-Directed) 估计验前信噪比 (Priori SNR)
        # ξ(k) = a * G_prev^2 * γ_prev + (1-a) * max(γ-1, 0)
        if self.is_first_frame:
            priori = self.a_dd + (1 - self.a_dd) * posteri_prime
            self.is_first_frame = False
        else:
            priori = self.a_dd * (self.G_prev**2) * self.posteri_prev + \
                     (1 - self.a_dd) * posteri_prime
        
        # 4. 计算 VAD (基于似然比的简单判断)
        log_sigma_k = posteri * priori / (1 + priori + self.eps) - np.log(1 + priori + self.eps)
        vad_decision = np.sum(log_sigma_k) / self.fft_bins
        
        # 5. 计算维纳增益函数
        # 注意：MATLAB 源码中使用了 sqrt(priori/(1+priori))，
        # 这实际上是针对幅度谱的维纳滤波增益。
        gain = np.sqrt(priori / (1 + priori + self.eps))
        
        # 更新状态供下一帧使用
        self.G_prev = gain
        self.posteri_prev = posteri
        
        return gain, vad_decision
