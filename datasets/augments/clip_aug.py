"""
@FileName: clip_aug.py
@Description: Implement clip_aug
@Author: Ryuk
@CreateDate: 2022/09/22
@LastEditTime: 2022/09/22
@LastEditors: Please set LastEditors
@Version: v0.1
"""

import random
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

class ClipTransform(nn.Module):
    def __init__(self, clip_ceil=1, clip_floor=0.5):
        super().__init__()
        self.clip_ceil = clip_ceil
        self.clip_floor = clip_floor

    def get_clip(self):
        return (self.clip_floor - self.clip_ceil) * random.random() + self.clip_ceil


    def forward(self, x):
        clip_level = self.get_clip()
        x[torch.abs(x)>clip_level] = clip_level
        return x


if __name__ == "__main__":
    speech, sr1 = torchaudio.load("../sample/speech.wav")
    noise, sr2 = torchaudio.load("../sample/noise.wav")

    assert sr1 == sr2
    assert speech.shape == noise.shape

    speech = torch.vstack((speech, speech))
    noise = torch.vstack((noise, noise))

    transformer = ClipTransform()
    s = transformer(speech)
    n = transformer(noise)

    sf.write("../sample/s_clip_trans.wav", s[0].numpy(), sr1)
    sf.write("../sample/n_clip_trans.wav", n[0].numpy(), sr2)