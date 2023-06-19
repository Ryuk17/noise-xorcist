"""
@FileName: mix_aug.py
@Description: Implement mix_aug
@Author: Ryuk
@CreateDate: 2022/09/19
@LastEditTime: 2022/09/19
@LastEditors: Please set LastEditors
@Version: v0.1
"""

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

class MixTransform(nn.Module):
    def __init__(self, snr_ceil=30, snr_floor=-5):
        super().__init__()
        self.snr_ceil = snr_ceil
        self.snr_floor = snr_floor

    def get_snr(self, n):
        return (self.snr_floor - self.snr_ceil) * torch.rand([n]) + self.snr_ceil

    def forward(self, speech, noise):
        samples = speech.size(0)
        snr = self.get_snr(samples)
        noise = noise  * torch.norm(speech) / torch.norm(noise)
        scalar = torch.pow(10.0, (0.05 * snr)).reshape([speech.size(0), 1])
        noise = torch.div(noise, scalar)
        mix = speech + noise
        return mix


if __name__ == "__main__":
    speech, sr1 = torchaudio.load("../sample/speech.wav")
    noise, sr2 = torchaudio.load("../sample/noise.wav")

    assert sr1 == sr2
    assert speech.shape == noise.shape

    speech = torch.vstack((speech, speech))
    noise = torch.vstack((noise, noise))

    transformer = MixTransform(-5,30)
    mix = transformer(speech, noise)

    sf.write("../sample/mix_trans.wav", mix[0].numpy(), sr1)
