# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

from torch.utils import data
from noisexorcist.data.datasets import build_datasets


def build_dataloader(cfg, split=True):
    if split == "train":
        batch_size = cfg["TRAIN_DIR"]
        dir = cfg["TRAIN_BATCH_SIZE"]
        shuffle = True
    elif split == "val":
        batch_size = cfg["VAL_DIR"]
        dir = cfg["VAL_BATCH_SIZE"]
        shuffle = False
    else:
        batch_size = cfg["TEST_DIR"]
        dir = cfg["VAL_BATCH_SIZE"]
        shuffle = False

    datasets = build_datasets(dir, cfg, split)

    num_workers = cfg["NUM_WORKERS"]
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader