"""
@FileName: __init__.py
@Description: Implement __init__
@Author: Ryuk
@CreateDate: 2022/09/20
@LastEditTime: 2022/09/20
@LastEditors: Please set LastEditors
@Version: v0.1
"""
import torch.distributed as dist
from torch.utils.data import SequentialSampler, DistributedSampler, RandomSampler

from .norm_freq_vad import NormFreqVAD

dataset_dict = {
    'norm_freq_vad': NormFreqVAD
}

def get_dataset(cfg, split):
    dataset_name = cfg['dataset']
    assert dataset_name in dataset_dict.keys(), f"There is no dataset named {dataset_name}"
    return dataset_dict[dataset_name](cfg, split)


def get_sampler(ddp, train_dataset, val_dataset):
    if not ddp:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    return train_sampler, SequentialSampler(val_dataset)

