'''
Author: Ryuk
Date: 2026-02-18 12:53:03
LastEditors: Ryuk
LastEditTime: 2026-02-18 12:56:20
Description: First create
'''

from ..base import BaseSpectralGainEstimator

import numpy as np
import scipy.io.wavfile as wav
from scipy.special import exp1

class LogMMSESpectralEstimator(BaseSpectralGainEstimator):
    """
    Ephraim, Y. and Malah, D. (1985). Speech enhancement using a minimum 
	mean-square error log-spectral amplitude estimator. IEEE Trans. Acoust., 
	Speech, Signal Process., ASSP-23(2), 443-445.
    """
    def __init__(self, n_fft, aa=0.98, mu=0.98, eta=0.15):
        super().__init__()
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1

        self.aa = aa           # Decision-Directed 因子
        self.mu = mu           # 噪声平滑因子
        self.eta = eta         # VAD 阈值
        self.ksi_min = 10**(-25/10)
        
        self.xk_prev = None    # 上一帧纯净信号功率
        self.is_first_frame = True

    def compute_gain(self, frame_fft, noise_mu2):
        """
        计算 Log-MMSE 增益
        """
        sig2 = np.abs(frame_fft)**2
        
        # 1. 计算验后信噪比 (Posteriori SNR)
        gammak = np.minimum(sig2 / (noise_mu2 + 1e-12), 40)
        
        # 2. 估计验前信噪比 (Priori SNR) - DD 法
        if self.is_first_frame:
            ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1, 0)
            self.is_first_frame = False
        else:
            # ksi = aa * (X_prev / Noise) + (1-aa) * max(gamma-1, 0)
            ksi = self.aa * (self.xk_prev / (noise_mu2 + 1e-12)) + \
                  (1 - self.aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(self.ksi_min, ksi)
            
        # 3. VAD 逻辑 (基于似然比累加)
        log_sigma_k = gammak * ksi / (1 + ksi + 1e-12) - np.log(1 + ksi + 1e-12)
        # 注意：MATLAB 代码这里用的是 /len，而非 /nFFT
        vad_decision = np.sum(log_sigma_k) / (self.n_fft // 2) 
        
        # 4. 计算 Log-MMSE 增益函数
        # 公式: G = (ksi/(1+ksi)) * exp(0.5 * E1( (ksi/(1+ksi)) * gammak ))
        A = ksi / (1 + ksi + 1e-12)
        vk = A * gammak
        
        # 处理 exp1(vk) 的数值问题，当 vk 非常小时 exp1 会趋向无穷
        # 我们给 vk 设置一个极小的下限以保证计算稳定
        ei_vk = 0.5 * exp1(np.maximum(vk, 1e-10))
        gain = A * np.exp(ei_vk)
        
        # 限制增益范围
        gain = np.clip(gain, 0.0, 1.0)
        
        # 5. 更新状态
        self.xk_prev = (gain**2) * sig2
        
        return gain, vad_decision, sig2

def process_logmmse(infile, outfile):
    fs, x = wav.read(infile)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
        
    frame_len = int(20 * fs / 1000)
    if frame_len % 2 == 1: frame_len += 1
    
    hop = frame_len // 2
    n_fft = 2 * frame_len
    
    # 窗函数归一化
    win = np.hanning(frame_len)
    win = win * hop / np.sum(win)
    
    estimator = LogMMSEEstimator(n_fft=n_fft)
    
    # 噪声初始化
    noise_mu2 = np.zeros(n_fft)
    for i in range(6):
        start = i * frame_len
        n_frame = x[start : start + frame_len] * win
        noise_mu2 += np.abs(np.fft.fft(n_frame, n_fft))**2
    noise_mu2 /= 6.0
    
    # 处理循环
    n_frames = (len(x) - frame_len) // hop - 1
    x_final = np.zeros(n_frames * hop + frame_len)
    x_old = np.zeros(hop)
    
    for n in range(n_frames):
        k = n * hop
        insign = x[k : k + frame_len] * win
        spec = np.fft.fft(insign, n_fft)
        
        gain, vad_val, sig2 = estimator.compute_gain(spec, noise_mu2)
        
        if vad_val < estimator.eta:
            noise_mu2 = estimator.mu * noise_mu2 + (1 - estimator.mu) * sig2
            
        enhanced_spec = gain * spec
        xi_w = np.real(np.fft.ifft(enhanced_spec, n_fft))
        
        x_final[k : k + hop] = x_old + xi_w[:hop]
        x_old = xi_w[hop : frame_len]
        
    wav.write(outfile, fs, (np.clip(x_final, -1, 1) * 32767).astype(np.int16))