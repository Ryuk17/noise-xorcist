# -*- coding:utf-8 -*-
"""
@author:：Ryuk
@contact: jeryuklau@gmail.com
"""

import torch


def build_window(win_type, win_len):
    if win_type == "hamming":
        return torch.hamming_window(win_len)
    elif win_type == "hanning":
        return torch.hanning_window(win_len)
    else:
        raise NotImplementedError
