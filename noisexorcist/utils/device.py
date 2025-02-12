# -*- coding:utf-8 -*-
"""
@author:：Ryuk
@contact: jeryuklau@gmail.com
"""

import torch


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(i, device) for i in data]
    elif isinstance(data, tuple):
        return tuple([to_device(i, device) for i in data])
    elif torch.is_tensor(data) and data.device != device:
        return data.to(device)
    else:
        return data

