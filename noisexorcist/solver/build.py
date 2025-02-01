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

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(cfg) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = copy.deepcopy(cfg)

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg["CLIP_GRADIENTS"]["CLIP_VALUE"], cfg["CLIP_GRADIENTS"]["NORM_TYPE"])

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg["CLIP_GRADIENTS"]["CLIP_VALUE"])

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg["CLIP_GRADIENTS"]["NORM_TYPE"])]


def _generate_optimizer_class_with_gradient_clipping(
        optimizer: Type[torch.optim.Optimizer],
        *,
        per_param_clipper: Optional[_GradientClipper] = None,
        global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert (
            per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    @torch.no_grad()
    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        optimizer.step(self, closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
        cfg, optimizer: Type[torch.optim.Optimizer]
) -> Type[torch.optim.Optimizer]:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.
    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer
    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    if not cfg["CLIP_GRADIENTS"]["ENABLED"]:
        return optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper(cfg["CLIP_GRADIENTS"])
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=grad_clipper
    )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip


def build_optimizer(cfg):
    if cfg["OPT"] == "SGD":
        return maybe_add_gradient_clipping(
            cfg, torch.optim.SGD
        )
    else:
        return maybe_add_gradient_clipping(cfg, getattr(torch.optim, cfg["OPT"]))


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
            "warmup_iters": cfg["SOLVER.WARMUP_ITERS"],
            "warmup_method": cfg["SOLVER.WARMUP_METHOD"],
        }
        scheduler_dict["warmup_sched"] = lr_scheduler.WarmupLR(**warmup_args)

    return scheduler_dict
