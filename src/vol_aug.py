"""
@FileName: vol_aug.py
@Description: Implement vol_aug
@Author: Ryuk
@CreateDate: 2022/09/20
@LastEditTime: 2022/09/20
@LastEditors: Please set LastEditors
@Version: v0.1
"""

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

class VolTransform(nn.Module):
    def __init__(self, sample_rate=16000, segment_len=0.5, vol_ceil=-10, vol_floor=10):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.segment_samples = int(self.sample_rate * self.segment_len)
        self.vol_ceil = vol_ceil
        self.vol_floor = vol_floor

    def get_vol(self, sample_length):
        segments = sample_length / (self.segment_len * self.sample_rate)
        step_db = torch.arange(self.vol_ceil, self.vol_floor, (self.vol_floor - self.vol_ceil) /segments)
        return step_db

    def apply_gain(self, segments, db):
        gain = torch.pow(10.0, (0.05 * db))
        segments = segments * gain
        return segments

    def forward(self, x):
        step_db = self.get_vol(x.size(1))
        for i in range(step_db.size(0)):
            start = i * self.segment_samples
            end = min((i+1) * self.segment_samples, x.size(1))
            x[:,start:end] = self.apply_gain(x[:, start:end], step_db[i])

        return x


if __name__ == "__main__":
    speech, sr1 = torchaudio.load("../sample/speech.wav")
    noise, sr2 = torchaudio.load("../sample/noise.wav")

    assert sr1 == sr2
    assert speech.shape == noise.shape

    speech = torch.vstack((speech, speech))
    noise = torch.vstack((noise, noise))

    transformer = VolTransform()
    s = transformer(speech)
    n = transformer(noise)

    sf.write("../sample/s_vol_trans.wav", s[0].numpy(), sr1)
    sf.write("../sample/n_vol_trans.wav", n[0].numpy(), sr2)
