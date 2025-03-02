# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging
import torch
from .nsnet import build_nsnet
from .crn import build_crn
from .dpcrn import build_dprn
from .gcrn import build_gcrn
from noisexorcist.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


logger = logging.getLogger(__name__)

models_dict = {
    "nsnet": build_nsnet,
    "crn": build_crn,
    "dprn": build_dprn,
    "gcrn": build_gcrn
}


def build_model(cfg):
    if cfg['NAME'] in models_dict:
        model = models_dict[cfg['NAME']](*cfg["PARAMS"])

        if cfg["PRETRAIN"]:
            try:
                state_dict = torch.load(cfg["PRETRAIN"], map_location=torch.device('cpu'))
                logger.info(f"Loading pretrained model from {cfg['PRETRAIN']}")
            except FileNotFoundError as e:
                logger.info(f'{cfg["PRETRAIN"]} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e

            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                logger.info(
                    get_missing_parameters_message(incompatible.missing_keys)
                )
            if incompatible.unexpected_keys:
                logger.info(
                    get_unexpected_parameters_message(incompatible.unexpected_keys)
                )

        model.to(torch.device(cfg["DEVICE"]))
        return model
    else:
        logger.error(f"Invalid model named {cfg['NAME']}")
        raise KeyError