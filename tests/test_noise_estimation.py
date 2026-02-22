'''
Author: Ryuk
Date: 2026-02-15 17:10:24
LastEditors: Ryuk
LastEditTime: 2026-02-22 18:48:50
Description: First create
'''


import sys
from pathlib import Path
from tqdm import tqdm

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from signal_processing.noise_estimation import (
    CFRNoiseEstimator,
    CSMTNoiseEstimator,
    IMCRANoiseEstimator,
    MCRANoiseEstimator,
    MCRA2NoiseEstimator,
    MSNoiseEstimator,
    SPPNoiseEstimator,
    WSANoiseEstimator
)

from signal_processing.spectral_gain_estimation import (
    SSSpectralGainEstimator,
)

import numpy as np
import librosa
import soundfile as sf


class TestNoiseEstimator:
    def __init__(self, noise_estimator, spectral_gain_estimator, n_fft=256, frame_len=256, hop_len=128, win_type="hamming", eps=1e-12):
        self.noise_estimator = noise_estimator
        self.spectral_gain_estimator = spectral_gain_estimator

        self.frame_len = frame_len
        self.hop_len = hop_len

        assert noise_estimator.n_fft == spectral_gain_estimator.n_fft == n_fft
        self.n_fft = n_fft
        self.fft_bins = self.n_fft // 2 + 1
        
        if win_type == "hamming":
            self.win = np.hamming(self.n_fft)
        else:
            raise NotImplementedError("Invalid window type.")
        
        self.eps = eps

    def process(self, data):
        # 1. 分帧
        frames = librosa.util.frame(data, frame_length=self.frame_len, hop_length=self.hop_len) 
        
        # 2. 准备输出数组（需要比原数据稍微长一点，防止最后一帧溢出）
        output_size = (frames.shape[1] - 1) * self.hop_len + self.frame_len
        processed_data = np.zeros(output_size)
        
        for i in tqdm(range(frames.shape[1])):
            frame = frames[:, i]
            
            # --- 频域处理 ---
            win_frame = frame * self.win # 分析窗
            spectrum = np.fft.rfft(win_frame, self.n_fft)
            
            frame_psd = np.abs(spectrum) ** 2
            noise_psd = self.noise_estimator.estimate_noise(frame_psd)
            gain = self.spectral_gain_estimator.compute_gain(frame_psd, noise_psd)
            
            # 应用增益
            processed_spectrum = spectrum * gain
            # 逆变换回时域
            processed_frame = np.fft.irfft(processed_spectrum, self.n_fft)
            
            # --- 关键修正：完整重叠累加 ---
            # 必须再次乘以窗函数（合成窗），以保证重叠处的平滑过渡
            processed_frame = processed_frame[:self.frame_len] * self.win 
            
            start = i * self.hop_len
            end = start + self.frame_len
            processed_data[start:end] += processed_frame
        
        return processed_data[:len(data)] 



if __name__ == "__main__":

    noise_estimator_dict = {
        "CFRNoiseEstimator": CFRNoiseEstimator,
        "CSMTNoiseEstimator": CSMTNoiseEstimator,
        "IMCRANoiseEstimator": IMCRANoiseEstimator,
        "MCRANoiseEstimator": MCRANoiseEstimator,
        "MCRA2NoiseEstimator": MCRA2NoiseEstimator,
        "MSNoiseEstimator": MSNoiseEstimator,
        "SPPNoiseEstimator": SPPNoiseEstimator,
        "WSANoiseEstimator": WSANoiseEstimator
    }

    n_fft = 256
    noisy_audio, sr = librosa.load("./samples/audio_long16noise.wav", sr=None)

    for noise_estimator_name in noise_estimator_dict:
        print(f"Testing {noise_estimator_name}...")
        noise_estimator = noise_estimator_dict[noise_estimator_name](n_fft=n_fft)
        spectral_gain_estimator = SSSpectralGainEstimator(n_fft=n_fft)
        
        denoiser = TestNoiseEstimator(noise_estimator, spectral_gain_estimator, n_fft=n_fft, frame_len=n_fft, hop_len=n_fft//2)
        
        enhanced_audio = denoiser.process(noisy_audio)
        
        sf.write(f"./samples/{noise_estimator_name}-{'SSSpectralGainEstimator'}.wav", enhanced_audio, sr)
