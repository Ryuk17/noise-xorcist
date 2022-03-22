"""
@FileName: RNNTransform.py
@Description: Implement RNNTransform
@Author: Ryuk
@CreateDate: 2022/03/01
@LastEditTime: 2022/03/01
@LastEditors: Please set LastEditors
@Version: v0.1
"""

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

class SpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_hp = torch.tensor([-1.99599, 0.99600])
        self.b_hp = torch.tensor([-2, 1])

    def _uni_rand(self):
        return torch.rand(1) - 0.5

    def _rand_resp(self):
        a1 = 0.75 * self._uni_rand()
        a2 = 0.75 * self._uni_rand()
        b1 = 0.75 * self._uni_rand()
        b2 = 0.75 * self._uni_rand()
        return a1, a2, b1, b2

    def forward(self, x, n):
        a1, a2, b1, b2 = self._rand_resp()
        x = torchaudio.functional.biquad(x, 1, self.b_hp[0], self.b_hp[1], 1, self.a_hp[0], self.a_hp[1])
        x = torchaudio.functional.biquad(x, 1, b1, b2, 1, a1, a2)

        a1, a2, b1, b2 = self._rand_resp()
        n = torchaudio.functional.biquad(n, 1, self.b_hp[0], self.b_hp[1], 1, self.a_hp[0], self.a_hp[1])
        n = torchaudio.functional.biquad(n, 1, b1, b2, 1, a1, a2)
        return x, n


if __name__ == "__main__":
    speech, sr1 = torchaudio.load("./sample/speech.wav")
    noise, sr2 = torchaudio.load("./sample/noise.wav")
    assert sr1 == sr2
    assert speech.shape == noise.shape

    speech = torch.vstack((speech, speech))
    noise = torch.vstack((noise, noise))

    transformer = SpecTransform()
    s, n = transformer(speech, noise)

    sf.write("./sample/s.wav", s[0].numpy(), sr1)
    sf.write("./sample/n.wav", n[0].numpy(), sr2)