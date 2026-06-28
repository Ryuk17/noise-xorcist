"""
DNS3 数据集 — 读取 DNS3 挑战赛的带噪/干净语音配对数据。
"""
from random import random
import soundfile as sf
import librosa
import torch
import numpy as np
import random

NOISY_DATABASE_TRAIN = '/data/ssd0/xiaobin.rong/Datasets/DNS3/train_noisy'
NOISY_DATABASE_VALID = '/data/ssd0/xiaobin.rong/Datasets/DNS3/dev_noisy'


class DNS3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=16000,
        length_in_seconds=8,
        num_data_tot=720000,
        num_data_per_epoch=40000,
        random_start_point=False,
        train=True
    ):
        if train:
            print("You are using this DNS3 training data:", NOISY_DATABASE_TRAIN)
        else:
            print("You are using this DNS3 validation data:", NOISY_DATABASE_VALID)
        self.noisy_database_train = sorted(librosa.util.find_files(NOISY_DATABASE_TRAIN, ext='wav'))[:num_data_tot]
        self.noisy_database_valid = sorted(librosa.util.find_files(NOISY_DATABASE_VALID, ext='wav'))
        self.L = int(length_in_seconds * fs)
        self.random_start_point = random_start_point
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train

    def sample_data_per_epoch(self):
        self.noisy_data_train = random.sample(self.noisy_database_train, self.num_data_per_epoch)

    def __getitem__(self, idx):
        if self.train:
            noisy_list = self.noisy_data_train
        else:
            noisy_list = self.noisy_database_valid

        if self.random_start_point:
            Begin_S = int(np.random.uniform(0, 10 - self.length_in_seconds)) * self.fs
            noisy, _ = sf.read(noisy_list[idx], dtype='float32', start=Begin_S, stop=Begin_S + self.L)
            clean, _ = sf.read(noisy_list[idx].replace('noisy', 'clean'), dtype='float32', start=Begin_S, stop=Begin_S + self.L)

        else:
            noisy, _ = sf.read(noisy_list[idx], dtype='float32', start=0, stop=self.L)
            clean, _ = sf.read(noisy_list[idx].replace('noisy', 'clean'), dtype='float32', start=0, stop=self.L)

        return noisy, clean

    def __len__(self):
        if self.train:
            return self.num_data_per_epoch
        else:
            return len(self.noisy_database_valid)
