"""
@FileName: howl_aug.py
@Description: Implement howl_aug
@Author: Ryuk
@CreateDate: 2022/09/26
@LastEditTime: 2022/09/26
@LastEditors: Please set LastEditors
@Version: v0.1
"""

from scipy import signal
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import random
import numpy as np

class HowlingTransform(nn.Module):
    def __init__(self, air, gain_floor=1, gain_ceil=10):
        super().__init__()
        self.air = air
        self.gain_floor = gain_floor
        self.gain_ceil = gain_ceil

    def get_MSG(self):
        ir_spec = torch.fft.rfft(self.air)
        ir_mag = torch.abs(ir_spec)
        ir_phase = torch.angle(ir_spec)

        MLG = torch.mean(torch.abs(ir_mag) ** 2)
        zero_phase_index = np.where(np.logical_and(-0.1 < ir_phase, ir_phase < 0.1))
        ir_zero_phase_mag = ir_mag[zero_phase_index]
        peak_gain = torch.max(torch.abs(ir_zero_phase_mag) ** 2)
        MSG = -10 * torch.log10(peak_gain / MLG)
        return MSG

    def get_gain(self):
        return (self.gain_floor - self.gain_ceil) * random.random() + self.gain_ceil

    def howling(self, gain):
        pass
    
    def forward(self, x):
        gain = self.get_MSG() + self.get_gain()

        return x


if __name__ == "__main__":
    speech, sr1 = torchaudio.load("../sample/speech.wav")
    air, sr2 = torchaudio.load("../sample/air.wav")
    assert sr1 == sr2

    speech = torch.vstack((speech, speech))
    transformer = HowlingTransform(air)
    s = transformer(speech)

    sf.write("../sample/s_howling_trans.wav", s[0].numpy(), sr1)

