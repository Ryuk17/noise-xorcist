# -*- coding:utf-8 -*-
"""
@author:：Ryuk
@contact: jeryuklau@gmail.com
"""

import torch
import logging
logger = logging.getLogger(__name__)


def replace_denormals(x: torch.tensor, threshold=1e-10):
    """ Returns a tensor without denormal values under a certain threshold.

    IMPORTANT: Please note that this function does not turn the denormal values
    into zeros, but replaces them by a threshold instead, thus preventing
    tensors with values that may turn into a NaN during backpropagation.

    Args:
        x (torch.tensor): Input tensor.
        threshold (float): Thereshold under or above which values will be set
            to its value to avoid zeros.
    """
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y