'''
Author: Ryuk
Date: 2026-02-15 17:10:24
LastEditors: Ryuk
LastEditTime: 2026-02-16 16:49:08
Description: First create
'''

import sys
from pathlib import Path

# 将当前文件的上两级目录（即项目根目录）加入搜索路径
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from base import BaseDenoiser
from noise_estimation.mcra import MCRANoiseEstimator
from spectral_gain_estimation.spectral_subtractive import BeroutiGainEstimator

import numpy as np
import librosa
import soundfile as sf
from scipy.fftpack import fft, ifft

class MCRASpectralSubtractive(BaseDenoiser):
    def __init__(self, noise_estimator, spectral_gain_estimator, frame_size=320, hop_size=160):
        super().__init__(noise_estimator, spectral_gain_estimator)
        self.frame_size = frame_size
        self.hop_size = hop_size

        assert noise_estimator.n_fft == spectral_gain_estimator.n_fft
        self.n_fft = noise_estimator.n_fft
        self.win = np.hamming(frame_size)
        
        # 用于存储重叠相加的状态（如果是流式处理，这很重要）
        self._prev_overlap = np.zeros(self.frame_size)

    def process_frame(self, frame_time_domain):
        """
        核心方法：按帧处理时域信号并返回处理后的时域帧
        
        参数:
            frame_time_domain (ndarray): 长度为 self.frame_size 的时域信号
        返回:
            enhanced_frame (ndarray): 降噪后的时域信号帧
        """
        # 1. 加窗与 FFT
        windowed_frame = frame_time_domain * self.win
        spec = fft(windowed_frame, self.n_fft)
        
        # 获取正频率部分 (0 到 fs/2)
        half_idx = self.n_fft // 2 + 1
        spec_half = spec[:half_idx]
        mag = np.abs(spec_half)
        phase = np.angle(spec_half)
        psd = mag ** 2
        
        # 2. 估计噪声 (调用组件)
        noise_estimate = self.noise_estimator.estimate_noise(psd)

        # 3. 计算谱增益 (调用组件)
        gain = self.spectral_gain.compute_gain(psd, noise_estimate)

        # 4. 应用增益
        # 这里的 _apply_gain 是你在基类中定义的，或者是为了处理复数谱重写的
        complex_spec_half = mag * np.exp(1j * phase)
        enhanced_spec_half = self._apply_gain(complex_spec_half, gain)
        
        # 5. 重构全谱并 IFFT
        # 拼接共轭对称部分以保证 IFFT 结果为实数
        enhanced_spec_full = np.concatenate([
            enhanced_spec_half, 
            np.conj(enhanced_spec_half[-2:0:-1])
        ])
        enhanced_frame = np.real(ifft(enhanced_spec_full))[:self.frame_size]
        
        return enhanced_frame

    def process(self, signal):
        """
        批处理方法：处理整段音频信号
        """
        num_samples = len(signal)
        num_frames = (num_samples - self.frame_size) // self.hop_size
        output = np.zeros(num_samples)
        
        # 重置流式状态（针对单次全量处理）
        self._prev_overlap.fill(0)
        
        for n in range(num_frames):
            idx = n * self.hop_size
            current_frame = signal[idx : idx + self.frame_size]
            
            # 调用新增的按帧处理函数
            enhanced_frame = self.process_frame(current_frame)
            
            # 重叠相加 (Overlap-Add)
            output[idx : idx + self.hop_size] = (
                self._prev_overlap[:self.hop_size] + enhanced_frame[:self.hop_size]
            )
            self._prev_overlap = enhanced_frame[self.hop_size:]
            
        return output



if __name__ == "__main__":
    # 加载音频
    noisy_audio, sr = librosa.load("../samples/audio_long16noise.wav", sr=None)
    
    # 初始化噪声估计器和谱增益计算器
    noise_estimator = MCRANoiseEstimator(n_fft=512)
    spectral_gain_estimator = BeroutiGainEstimator(n_fft=512)
    
    # 初始化降噪算法
    denoiser = MCRASpectralSubtractive(noise_estimator, spectral_gain_estimator, frame_size=512, hop_size=256)
    
    # 处理音频
    enhanced_audio = denoiser.process(noisy_audio)
    
    # 保存结果
    sf.write("../samples/MCRASpectralSubtractive.wav", enhanced_audio, sr)
