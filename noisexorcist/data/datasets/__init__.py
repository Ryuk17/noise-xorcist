# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

from spectrums import Spectrum



__all__ = {
    'spectrum': Spectrum
}


def build_dataset(cfg, is_train, verbose=False, **kwargs):
    if is_train:
        mode = "train"
    else:
        mode = "test"

    dataset = cfg['DATA']['DATASET']
    return __all__[dataset](cfg, mode, verbose, **kwargs)

