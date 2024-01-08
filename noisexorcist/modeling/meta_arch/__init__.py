# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import torch

from .baseline import Baseline
from .distiller import Distiller

__all__ = {
    "Baseline": Baseline,
    "Distiller": Distiller
}

def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = __all__[meta_arch](cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
