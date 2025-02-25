# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging
from .nsnet import build_nsnet
from .crn import build_crnnet


logger = logging.getLogger(__name__)

models_dict = {
    "nsnet": build_nsnet,
    "crnnet": build_crnnet
}


def build_model(cfg):
    if cfg['NAME'] in models_dict:
        return models_dict[cfg['NAME']](*cfg["PARAMS"])
    else:
        logger.error(f"Invalid model named {cfg['NAME']}")
        raise KeyError