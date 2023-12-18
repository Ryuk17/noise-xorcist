"""
@FileName: filter_aug.py
@Description: Implement filter_aug
@Author: Ryuk
@CreateDate: 2022/09/20
@LastEditTime: 2022/09/20
@LastEditors: Please set LastEditors
@Version: v0.1
"""

import random
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

class FilterTransform(nn.Module):
    def __init__(self, sample_rate=16000, freq_ceil=8000, freq_floor=0, gain_ceil=20, gain_floor=-20, Q=0.707):
        super().__init__()
        self.sample_rate = sample_rate
        self.freq_ceil = freq_ceil
        self.freq_floor = freq_floor
        self.gain_ceil = gain_ceil
        self.gain_floor = gain_floor
        self.Q = Q

    def get_gain(self):
        return (self.gain_floor - self.gain_ceil) * random.random() + self.gain_ceil

    def get_center_freq(self):
        return (self.freq_floor - self.freq_ceil) * random.random() + self.freq_ceil


    def forward(self, x):
        gain = self.get_gain()
        center_freq = self.get_center_freq()

        x = torchaudio.functional.equalizer_biquad(x,
                                                   sample_rate=self.sample_rate,
                                                   center_freq=center_freq,
                                                   gain=gain,
                                                   Q=self.Q)
        return x


if __name__ == "__main__":
    speech, sr1 = torchaudio.load("../sample/speech.wav")
    noise, sr2 = torchaudio.load("../sample/noise.wav")

    assert sr1 == sr2
    assert speech.shape == noise.shape

    speech = torch.vstack((speech, speech))
    noise = torch.vstack((noise, noise))

    transformer = FilterTransform()
    s = transformer(speech)
    n = transformer(noise)

    sf.write("../sample/s_filter_trans.wav", s[0].numpy(), sr1)
    sf.write("../sample/n_filter_trans.wav", n[0].numpy(), sr2)