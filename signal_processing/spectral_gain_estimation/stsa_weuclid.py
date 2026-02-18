'''
Author: Ryuk
Date: 2026-02-18 12:53:55
LastEditors: Ryuk
LastEditTime: 2026-02-18 13:37:22
Description: First create
'''

from base import BaseSpectralGainEstimator

import numpy as np
import scipy.io.wavfile as wav
from scipy.special import gamma, i0, hyp1f1

class STSAWeuclidSpectralGainEstimator(BaseSpectralGainEstimator):
    """
        Loizou, P. (2005). Speech enhancement based on perceptually motivated 
        Bayesian estimators of the speech magnitude spectrum. IEEE Trans. on Speech 
        and Audio Processing, 13(5), 857-869.
    """
    def __init__(self, n_fft, p=-1.0, aa=0.98, mu=0.98, eta=0.15):
        super().__init__()
        if p <= -2:
            raise ValueError("p 必须大于 -2")
            
        self.n_fft = n_fft
        self.p = p
        self.aa = aa
        self.mu = mu
        self.eta = eta
        self.ksi_min = 10**(-25/10)
        
        # 预计算常数 CC
        # CC = gamma((p+3)/2) / gamma(p/2+1)
        self.CC = gamma((p + 3) / 2) / gamma(p / 2 + 1)
        
        self.xk_prev = None
        self.is_first_frame = True

    def compute_gain(self, frame_fft, noise_mu2):
        sig2 = np.abs(frame_fft)**2
        # 1. 验后信噪比
        gammak = np.minimum(sig2 / (noise_mu2 + 1e-12), 40.0)
        
        # 2. 验前信噪比 (Decision-Directed)
        if self.is_first_frame:
            ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1, 0)
            self.is_first_frame = False
        else:
            ksi = self.aa * self.xk_prev / (noise_mu2 + 1e-12) + \
                  (1 - self.aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(self.ksi_min, ksi)
            
        # 3. VAD
        log_sigma_k = gammak * ksi / (1 + ksi + 1e-12) - np.log(1 + ksi + 1e-12)
        vad_decision = np.sum(log_sigma_k) / (self.n_fft // 2) # 对应 MATLAB 的 len
        
        # 4. 计算增益 hw
        vk = ksi * gammak / (1 + ksi + 1e-12)
        
        if self.p == -1:
            # 当 p = -1 时的快速简化公式
            # hw = CC * sqrt(vk) / (gammak * exp(-vk/2) * I0(vk/2))
            # 注意：MATLAB 代码中用了 besseli(0, vk/2)
            # 为了稳定，我们使用带有 exp 补偿的 i0e
            import scipy.special as sp
            denom = gammak * sp.i0e(vk / 2.0)
            hw = self.CC * np.sqrt(vk + 1e-12) / (denom + 1e-12)
        else:
            # 通用情况使用合流超几何函数 hyp1f1(a, b, x)
            # numer = CC * sqrt(vk) * hyp1f1(-(p+1)/2, 1, -vk)
            # denom = gammak * hyp1f1(-p/2, 1, -vk)
            numer = self.CC * np.sqrt(vk + 1e-12) * hyp1f1(-(self.p + 1) / 2, 1, -vk)
            denom = gammak * hyp1f1(-self.p / 2, 1, -vk)
            hw = numer / (denom + 1e-12)
            
        # 增益限制
        hw = np.nan_to_num(hw, nan=0.0)
        hw = np.clip(hw, 0.0, 1.0)
        
        # 状态更新
        self.xk_prev = (hw * np.abs(frame_fft))**2
        
        return hw, vad_decision, sig2

def stsa_weuclid_process(infile, outfile, p=-1.0):
    fs, x = wav.read(infile)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
        
    frame_len = int(20 * fs / 1000)
    if frame_len % 2 == 1: frame_len += 1
    hop = frame_len // 2
    n_fft = 2 * frame_len
    
    # 窗函数归一化 (MATLAB: win * len2 / sum(win))
    win = np.hanning(frame_len)
    win = win * hop / np.sum(win)
    
    estimator = WeuclidEstimator(n_fft=n_fft, p=p)
    
    # 噪声初始化
    noise_mu2 = np.zeros(n_fft)
    for i in range(6):
        start = i * frame_len
        n_frame = x[start : start + frame_len] * win
        noise_mu2 += np.abs(np.fft.fft(n_frame, n_fft))**2
    noise_mu2 /= 6.0
    
    # 主循环
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
        
        # 叠加
        x_final[k : k + hop] = x_old + xi_w[:hop]
        x_old = xi_w[hop : frame_len]
        
    wav.write(outfile, fs, (np.clip(x_final, -1, 1) * 32767).astype(np.int16))