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

class FilterTransform(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.a_hp = [-1.99599, 0.99600]
        self.b_hp = [-2, 1]
        self.mem_hp_x = torch.zeros(2)
        self.mem_hp_n = torch.zeros(2)
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
        y = torch.tensor(x.shape)
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

        n = self._biquad(self.mem_hp_n, n, self.b_hp, self.a_hp)
        n = self._biquad(self.mem_resp_n, n, b, a)
        return x, n