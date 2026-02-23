'''
Author: Ryuk
Date: 2026-02-18 12:53:23
LastEditors: Ryuk
LastEditTime: 2026-02-23 22:13:58
Description: First create
'''

from ..base import BaseSpectralGainEstimator
import numpy as np
from scipy.special import exp1, iv
from scipy.signal import lfilter, windows


class LogMMSESpuSpectralEstimator(BaseSpectralGainEstimator):
    """
        Cohen, I. (2002). Optimal speech enhancement under signal presence 
        uncertainty using log-spectra amplitude estimator. IEEE Signal Processing 
        Letters, 9(4), 113-116.
    """
    def __init__(self, n_fft, option=4, eps=1e-12):
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1
        self.option = option
        
        # 常量设置
        self.aa = 0.98
        self.mu = 0.98
        self.eta = 0.15
        self.ksi_min = 10**(-25/10)
        self.ksi_max = 10**(20/10)
        self.Gmin = 10**(-10/10) 
        
        # 状态变量
        self.xk_prev = None
        self.ksi_old = np.zeros(self.fft_bins)
        self.qk = 0.5 * np.ones(self.fft_bins)
        self.is_first_frame = True
        
        # Cohen (2002) 专有状态 (Option 4)
        if self.option == 4:
            self.zetak = np.zeros(self.fft_bins)
            self.zeta_fr_old = 1000.0
            self.z_peak = 0.0
        
        self.eps = eps

    def _smoothing(self, x, N):
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

    def _est_sap(self, qk, ksi, ksi_old, gammak):
        """估计先验语音缺失概率 qk"""
        if self.option == 1: # Hard-decision (Soon et al.)
            beta = 0.1
            dk = np.ones(len(ksi))
            # i0(2*sqrt(gamma*ksi))
            i0 = iv(0, 2 * np.sqrt(np.maximum(gammak * ksi, 0)))
            temp = np.exp(-ksi) * i0
            dk[temp > 1] = 0
            qk = beta * dk + (1 - beta) * qk
            
        elif self.option == 2: # Soft-decision (Soon et al.)
            beta = 0.1
            i0 = iv(0, 2 * np.sqrt(np.maximum(gammak * ksi, 0)))
            temp = np.exp(-ksi) * i0
            p_ho = 1.0 / (1.0 + temp + self.eps)
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
            len2 = self.fft_bins
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
            qk = beta * qk + (1 - beta) * qk2
        return qk

    def compute_gain(self, frame_psd, noise_psd):
        gammak = np.minimum(frame_psd / (noise_psd + self.eps), 40.0)
        
        if self.is_first_frame:
            ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1, 0)
            self.is_first_frame = False
        else:
            ksi = self.aa * self.xk_prev / (noise_psd + self.eps) + (1 - self.aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(self.ksi_min, ksi)
            ksi = np.minimum(self.ksi_max, ksi)
            
        # VAD
        log_sigma_k = gammak * ksi / (1 + ksi + self.eps) - np.log(1 + ksi + self.eps)
        vad_decision = np.sum(log_sigma_k) / self.fft_bins
        
        # Log-MMSE 基础增益 (hw)
        A = ksi / (1 + ksi + self.eps)
        vk = A * gammak
        ei_vk = 0.5 * exp1(np.maximum(vk, self.eps))
        hw = A * np.exp(ei_vk)

        # SPU 概率计算
        self.qk = self._est_sap(self.qk, ksi, self.ksi_old, gammak)
        # P(H1 | Yk)
        p_sap = (1 - self.qk) / (1 - self.qk + self.qk * (1 + ksi) * np.exp(-vk) + self.eps)
        
        # Cohen (2002) 增益修正 (Eq 8)
        Gmin2 = self.Gmin ** (1 - p_sap)
        gain = (hw ** p_sap) * Gmin2
        
        # 状态更新
        self.xk_prev = (gain**2) * frame_psd
        self.ksi_old = ksi
        
        return gain, vad_decision
