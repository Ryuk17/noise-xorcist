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

class FilterTransform(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.a_hp = torch.tensor([-1.99599, 0.99600])
        self.b_hp = torch.tensor([-2, 1])
        self.mem_hp_x = torch.zeros(batch_size, 2)
        self.mem_hp_n = torch.zeros(batch_size, 2)
        self.mem_resp_x = torch.zeros((batch_size, 2))
        self.mem_resp_n = torch.zeros((batch_size, 2))
        self.batch_size = batch_size

    def _uni_rand(self):
        return torch.rand(self.batch_size) - 0.5

    def _rand_resp(self):
        a = 0.75 * self._uni_rand()
        b = 0.75 * self._uni_rand()
        return a, b

    def _biquad(self, mem, x, b, a):
        y = torch.zeros(x.shape)
        for i in range(x.shape[1]):
            xi = x[:, i]
            yi = x[:, i] + mem[:, 0]
            mem[:, 0] = mem[:, 1] + (b[0] * xi - a[0] * yi)
            mem[:, 1] = (b[1] * xi - a[1] * yi)
            y[:, i] = yi
        return y

    def forward(self, x, n):
        a, b = self._rand_resp()
        x = self._biquad(self.mem_hp_x, x, self.b_hp, self.a_hp)
        x = self._biquad(self.mem_resp_x, x, b, a)
        
        a_noise, b_noise = self._rand_resp()
        n = self._biquad(self.mem_hp_n, n, self.b_hp, self.a_hp)
        n = self._biquad(self.mem_resp_n, n, b_noise, a_noise)
        return x, n


if __name__ == "__main__":
    speech, sr1 = torchaudio.load("./sample/speech.wav")
    noise, sr2 = torchaudio.load("./sample/noise.wav")
    assert sr1 == sr2
    assert speech.shape == noise.shape

    speech = torch.vstack((speech, speech))
    noise = torch.vstack((noise, noise))

    transformer = FilterTransform(2)
    s, n = transformer(speech, noise)

    sf.write("./sample/s.wav", s[0].numpy(), sr1)
    sf.write("./sample/n.wav", n[0].numpy(), sr2)
