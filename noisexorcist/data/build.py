# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging
import os

import torch
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import string_classes
else:
    string_classes = str


from collections import Mapping
from noisexorcist.config import configurable
from noisexorcist.utils import comm
from .datasets import build_dataset
from . import samplers
from .data_utils import DataLoaderX

__all__ = [
    "build_se_train_loader",
    "build_se_test_loader"
]


def _train_loader_from_config(cfg, *, train_set=None, sampler=None, **kwargs):
    if train_set is None:
        train_set = build_dataset(cfg, True, **kwargs)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        mini_batch_size = cfg.SOLVER.BATCH_SIZE // comm.get_world_size()

        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = samplers.TrainingSampler(len(train_set))
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "train_set": train_set,
        "sampler": sampler,
        "total_batch_size": cfg.SOLVER.BATCH_SIZE,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_se_train_loader(
        train_set, *, sampler=None, total_batch_size, num_workers=0,
):
    """
    Build a dataloader for object speech enhancement with some default features.
    This interface is experimental.

    Returns:
        torch.utils.data.DataLoader: a dataloader.
    """

    mini_batch_size = total_batch_size // comm.get_world_size()

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, mini_batch_size, True)

    train_loader = DataLoaderX(
        comm.get_local_rank(),
        dataset=train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return train_loader


def _test_loader_from_config(cfg, *, test_set=None, **kwargs):
    if test_set is None:
        test_set = build_dataset(cfg, False, **kwargs)

    return {
        "test_set": test_set,
        "test_batch_size": cfg.TEST.IMS_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_test_loader_from_config)
def build_se_test_loader(test_set, test_batch_size, num_workers=4):
    """
    Similar to `build_se_train_loader`. This sampler coordinates all workers to produce
    the exact set of all samples
    This interface is experimental.

    Args:
        test_set:
        test_batch_size:

    Returns:
        DataLoader: a torch DataLoader, that loads the given se dataset, with
        the test-time transformation.

    Examples:
    ::
        data_loader = build_reid_test_loader(test_set, test_batch_size, num_query)
        # or, instantiate with a CfgNode:
        data_loader = build_reid_test_loader(cfg, "my_test")
    """

    mini_batch_size = test_batch_size // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoaderX(
        comm.get_local_rank(),
        dataset=test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return test_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common se tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs