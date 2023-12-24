# -*- coding:utf-8 -*-
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import numpy as np
import torch

class VadDetector:
    def __init__(self,  sample_rate, nfft, mode="freq", **kwargs):
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.mode = mode

        assert self.sample_rate in [8000, 16000]

        # VAD
        if self.mode == "freq":
            step = self.sample_rate / self.nfft
            frequency_bins = np.arange(0, (self.nfft // 2 + 1) * step, step=step)
            self.vad_frequencies = np.where((frequency_bins >= 300) & (frequency_bins <= 5000), True, False)
        elif self.mode == "webrtc":
            pass
        else:
            raise NotImplementedError

    def __call__(self, input, **kwargs):
        if self.mode == "freq":
            # VAD
            y_ms_filtered = input[self.vad_frequencies]
            y_energy_filtered = y_ms_filtered.pow(2).mean(dim=0)
            y_energy_filtered_averaged = self.__moving_average(y_energy_filtered)
            y_peak_energy = y_energy_filtered_averaged.max()
            VAD = torch.where(y_energy_filtered_averaged > y_peak_energy / 1000, torch.ones_like(y_energy_filtered), torch.zeros_like(y_energy_filtered))
            VAD = VAD.bool()
        elif self.mode == "webrtc":
            pass
        else:
            raise NotImplementedError

        return VAD

    def __moving_average(self, a, n=3):
        ret = torch.cumsum(a, dim=0)
        ret[n:] = ret[n:] - ret[:-n]
        ret[:n - 1] = a[:n - 1]
        ret[n - 1:] = ret[n - 1:] / n
        return ret
