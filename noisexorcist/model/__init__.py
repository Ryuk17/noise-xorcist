# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging
from .nsnet import build_nsnet


logger = logging.getLogger(__name__)

models_dict = {
    "NSNET": build_nsnet
}


def build_model(cfg):
    if cfg['NAME'] in models_dict:
        return models_dict[cfg['NAME']](cfg)
    else:
        logger.error(f"Invalid model named {cfg['NAME']}")
        raise KeyError