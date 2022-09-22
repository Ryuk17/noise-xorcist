"""
@FileName: reverb_aug.py
@Description: Implement reverb_aug
@Author: Ryuk
@CreateDate: 2022/09/22
@LastEditTime: 2022/09/22
@LastEditors: Please set LastEditors
@Version: v0.1
"""

from scipy import signal
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

class ReverbTransform(nn.Module):
    def __init__(self, rir):
        super().__init__()
        self.rir = rir

    def forward(self, x):
        reverbed = signal.fftconvolve(x, self.rir, mode="full")
        reverbed = torch.tensor(reverbed[0: x.shape[0]])
        return reverbed


if __name__ == "__main__":
    speech, sr1 = torchaudio.load("../sample/speech.wav")
    noise, sr2 = torchaudio.load("../sample/noise.wav")
    rir, sr3 = torchaudio.load("../sample/rir.wav")
    assert sr1 == sr2 == sr3
    assert speech.shape == noise.shape

    speech = torch.vstack((speech, speech))
    noise = torch.vstack((noise, noise))

    transformer = ReverbTransform(rir)
    s = transformer(speech)
    n = transformer(noise)

    sf.write("../sample/s_reverb_trans.wav", s[0].numpy(), sr1)
    sf.write("../sample/n_reverb_trans.wav", n[0].numpy(), sr2)