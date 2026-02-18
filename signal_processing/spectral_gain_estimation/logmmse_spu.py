'''
Author: Ryuk
Date: 2026-02-18 12:53:23
LastEditors: Ryuk
LastEditTime: 2026-02-18 13:01:15
Description: First create
'''

from base import BaseSpectralGainEstimator
import numpy as np
import scipy.io.wavfile as wav
from scipy.special import exp1, i0
from scipy.signal import lfilter, windows


class LogMMSESpuSpectralEstimator(BaseSpectralGainEstimator):
    """
    Cohen, I. (2002). Optimal speech enhancement under signal presence 
	uncertainty using log-spectra amplitude estimator. IEEE Signal Processing 
	Letters, 9(4), 113-116.
    """
    def __init__(self, fs, frame_len, option=4):
        self.fs = fs
        self.len = frame_len
        self.n_fft = frame_len
        self.option = option
        
        # 常量设置
        self.aa = 0.98
        self.mu = 0.98
        self.eta = 0.15
        self.ksi_min = 10**(-25/10)
        self.Gmin = 10**(-20/10)
        
        # 状态变量
        self.xk_prev = None
        self.ksi_old = np.zeros(self.n_fft)
        self.qk = 0.5 * np.ones(self.n_fft)
        self.is_first_frame = True
        
        # Cohen (2002) 专有状态 (Option 4)
        if self.option == 4:
            self.len2a = self.n_fft // 2 + 1
            self.zetak = np.zeros(self.len2a)
            self.zeta_fr_old = 1000.0
            self.z_peak = 0.0

    def _smoothing(self, x, N):
        """对应 MATLAB 中的 smoothing 函数"""
        length = len(x)
        win = windows.hann(2 * N + 1, sym=True)
        win1 = win[:N + 1]
        win2 = win[N + 1:]
        
        # y1 = filter(flipud(win1), [1], x)
        y1 = lfilter(np.flip(win1), [1], x)
        
        # y2 处理逻辑
        x2 = np.zeros(length)
        if length > N:
            x2[:length - N] = x[N:]
        y2 = lfilter(np.flip(win2), [1], x2)
        
        return (y1 + y2) / np.linalg.norm(win, 2)

    def _est_sap(self, ksi, ksi_old, gammak):
        """估计先验语音缺失概率 qk"""
        qk = self.qk
        
        if self.option == 1: # Hard-decision (Soon et al.)
            beta = 0.1
            dk = np.ones(len(ksi))
            # i0(2*sqrt(gamma*ksi))
            temp = np.exp(-ksi) * i0(2 * np.sqrt(np.maximum(gammak * ksi, 0)))
            dk[temp > 1] = 0
            qk = beta * dk + (1 - beta) * qk
            
        elif self.option == 2: # Soft-decision (Soon et al.)
            beta = 0.1
            temp = np.exp(-ksi) * i0(2 * np.sqrt(np.maximum(gammak * ksi, 0)))
            p_ho = 1.0 / (1.0 + temp + 1e-12)
            p_ho = np.minimum(1.0, p_ho)
            qk = beta * p_ho + (1 - beta) * qk
            
        elif self.option == 3: # Malah et al. (1999)
            if np.mean(gammak[:len(gammak)//2]) > 2.4:
                beta, gamma_th = 0.95, 0.8
                dk = np.ones(len(ksi))
                dk[gammak > gamma_th] = 0
                qk = beta * qk + (1 - beta) * dk
                
        elif self.option == 4: # Cohen (2002)
            beta = 0.7
            len2 = self.len2a
            # 更新 zetak
            self.zetak = beta * self.zetak + (1 - beta) * ksi_old[:len2]
            
            z_min, z_max = 0.1, 0.3162
            C = np.log10(z_max / z_min)
            zp_min, zp_max = 1.0, 10.0
            
            zeta_local = self._smoothing(self.zetak, 1)
            zeta_global = self._smoothing(self.zetak, 15)
            
            # P_local
            p_local = np.zeros(len2)
            p_local[zeta_local > z_max] = 1.0
            idx = (zeta_local > z_min) & (zeta_local < z_max)
            p_local[idx] = np.log10(zeta_local[idx] / z_min) / C
            
            # P_global
            p_global = np.zeros(len2)
            p_global[zeta_global > z_max] = 1.0
            idx = (zeta_global > z_min) & (zeta_global < z_max)
            p_global[idx] = np.log10(zeta_global[idx] / z_min) / C
            
            # P_frame
            zeta_fr = np.mean(self.zetak)
            p_frame = 0.0
            if zeta_fr > z_min:
                if zeta_fr > self.zeta_fr_old:
                    p_frame = 1.0
                    self.z_peak = np.minimum(np.maximum(zeta_fr, zp_min), zp_max)
                else:
                    if zeta_fr <= self.z_peak * z_min: p_frame = 0.0
                    elif zeta_fr >= self.z_peak * z_max: p_frame = 1.0
                    else: p_frame = np.log10(zeta_fr / self.z_peak / z_min) / C
            
            self.zeta_fr_old = zeta_fr
            qk2 = 1.0 - p_local * p_global * p_frame
            qk2 = np.minimum(0.95, qk2)
            # 对称化处理
            qk = np.concatenate([qk2, np.flip(qk2[1:-1])])
            
        self.qk = qk
        return qk

    def compute_gain(self, frame_fft, noise_mu2):
        sig2 = np.abs(frame_fft)**2
        gammak = np.minimum(sig2 / (noise_mu2 + 1e-12), 40.0)
        
        if self.is_first_frame:
            ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1, 0)
            self.is_first_frame = False
        else:
            ksi = self.aa * self.xk_prev / (noise_mu2 + 1e-12) + (1 - self.aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(self.ksi_min, ksi)
            
        # VAD
        log_sigma_k = gammak * ksi / (1 + ksi + 1e-12) - np.log(1 + ksi + 1e-12)
        vad_decision = np.sum(log_sigma_k) / self.len
        
        # Log-MMSE 基础增益 (hw)
        A = ksi / (1 + ksi + 1e-12)
        vk = A * gammak
        hw = A * np.exp(0.5 * exp1(np.maximum(vk, 1e-10)))
        
        # SPU 概率计算
        qk = self._est_sap(ksi, self.ksi_old, gammak)
        # P(H1 | Yk)
        p_sap = (1 - qk) / (1 - qk + qk * (1 + ksi) * np.exp(-vk) + 1e-12)
        
        # Cohen (2002) 增益修正 (Eq 8)
        g_cohen = (hw ** p_sap) * (self.Gmin ** (1 - p_sap))
        
        # 状态更新
        self.xk_prev = (g_cohen * np.abs(frame_fft))**2
        self.ksi_old = ksi
        
        return g_cohen, vad_decision, sig2

def logmmse_spu_process(infile, outfile, option=4):
    fs, x = wav.read(infile)
    if x.dtype == np.int16: x = x.astype(np.float32) / 32768.0
    
    frame_len = int(20 * fs / 1000)
    if frame_len % 2 == 1: frame_len += 1
    hop = frame_len // 2
    
    win = np.hamming(frame_len)
    estimator = LogMmseSPUEstimator(fs, frame_len, option)
    
    # 初始噪声
    noise_mu2 = np.zeros(frame_len)
    for i in range(6):
        n_frame = x[i*frame_len : (i+1)*frame_len] * win
        noise_mu2 += np.abs(np.fft.fft(n_frame, frame_len))**2
    noise_mu2 /= 6.0
    
    n_frames = (len(x) - frame_len) // hop - 1
    x_final = np.zeros(n_frames * hop + frame_len)
    x_old = np.zeros(hop)
    
    for n in range(n_frames):
        k = n * hop
        insign = x[k : k + frame_len] * win
        spec = np.fft.fft(insign, frame_len)
        
        gain, vad_val, sig2 = estimator.compute_gain(spec, noise_mu2)
        
        if vad_val < estimator.eta:
            noise_mu2 = estimator.mu * noise_mu2 + (1 - estimator.mu) * sig2
            
        xi_w = np.real(np.fft.ifft(gain * spec, frame_len))
        x_final[k : k + hop] = x_old + xi_w[:hop]
        x_old = xi_w[hop:]
        
    wav.write(outfile, fs, (np.clip(x_final, -1, 1) * 32767).astype(np.int16))