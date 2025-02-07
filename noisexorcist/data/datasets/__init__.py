# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging
from pathlib import Path
from .spectrum import SpectrumDataset

logger = logging.getLogger(__name__)

datasets_dict = {
    "spectrum": SpectrumDataset
}


def build_datasets(dir, cfg, split):
    if cfg['FEATURE'] in datasets_dict:
        return datasets_dict[cfg['FEATURE']](Path(dir), cfg, split)
    else:
        logger.error(f"Invalid feature named {cfg['FEATURE']}")
        raise KeyError