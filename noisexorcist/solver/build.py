# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

# Based on: https://github.com/facebookresearch/detectron2/blob/master/detectron2/solver/build.py

import copy
import itertools
import math
import re
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

import torch

from . import lr_scheduler


def build_optimizer(model, cfg):
    if cfg["OPT"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["BASE_LR"], momentum=cfg["MOMENTUM"], nesterov=cfg["NESTEROV"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["BASE_LR"])

    return optimizer


def build_lr_scheduler(cfg, optimizer, iters_per_epoch):
    max_epoch = cfg["MAX_EPOCH"] - max(
        math.ceil(cfg["WARMUP_ITERS"] / iters_per_epoch), cfg["DELAY_EPOCHS"])

    scheduler_dict = {}

    scheduler_args = {
        "MultiStepLR": {
            "optimizer": optimizer,
            # multi-step lr scheduler options
            "milestones": cfg["STEPS"],
            "gamma": cfg["GAMMA"],
        },
        "CosineAnnealingLR": {
            "optimizer": optimizer,
            # cosine annealing lr scheduler options
            "T_max": max_epoch,
            "eta_min": cfg["ETA_MIN_LR"],
        },

    }

    scheduler_dict["lr_sched"] = getattr(lr_scheduler, cfg["SCHED"])(
        **scheduler_args[cfg["SCHED"]])

    if cfg["WARMUP_ITERS"] > 0:
        warmup_args = {
            "optimizer": optimizer,

            # warmup options
            "warmup_factor": cfg["WARMUP_FACTOR"],
            "warmup_iters": cfg["WARMUP_ITERS"],
            "warmup_method": cfg["WARMUP_METHOD"],
        }
        scheduler_dict["warmup_sched"] = lr_scheduler.WarmupLR(**warmup_args)

    return scheduler_dict
