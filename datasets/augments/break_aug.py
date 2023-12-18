"""
@FileName: break_aug.py
@Description: Implement break_aug
@Author: Ryuk
@CreateDate: 2022/09/26
@LastEditTime: 2022/09/26
@LastEditors: Please set LastEditors
@Version: v0.1
"""

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import random

class BreakTransform(nn.Module):
    def __init__(self, sample_rate=16000, break_duration=0.01, break_ceil=50, break_floor=10):
        super().__init__()
        self.sample_rate = sample_rate
        self.break_segment = sample_rate * break_duration
        self.break_ceil = break_ceil
        self.break_floor = break_floor

    def get_mask(self, x):
        break_count = (self.break_floor - self.break_ceil)  * random.random() + self.break_ceil
        break_duration = break_count * self.break_segment
        mask = torch.ones(x.size())
        break_start = int(x.size(1) * random.random())
        break_end = int(min(x.size(1), break_start+break_duration))
        mask[:, break_start:break_end] = 0
        return mask

    def forward(self, x):
        break_mask = self.get_mask(x)
        x = x * break_mask
        return x


if __name__ == "__main__":
    speech, sr1 = torchaudio.load("../sample/speech.wav")
    noise, sr2 = torchaudio.load("../sample/noise.wav")

    assert sr1 == sr2
    assert speech.shape == noise.shape

    speech = torch.vstack((speech, speech))
    noise = torch.vstack((noise, noise))

    transformer = BreakTransform()
    s = transformer(speech)
    n = transformer(noise)

    sf.write("../sample/s_break_trans.wav", s[0].numpy(), sr1)
    sf.write("../sample/n_break_trans.wav", n[0].numpy(), sr2)
