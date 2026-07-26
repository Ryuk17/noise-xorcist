"""
VoiceBank-DEMAND 16kHz dataset — 通过 scp 文件指定样本 ID 列表，配合 noisy_dir/clean_dir 拼接路径。

scp 文件格式（每行一个 audio id）：
    p226_001
    p226_002

实际路径拼接：
    {noisy_dir}/{id}.wav   →  /input0/noisy_trainset_28spk_wav/p226_001.wav
    {clean_dir}/{id}.wav   →  /input0/clean_trainset_28spk_wav/p226_001.wav
"""
import random

import numpy as np
import soundfile as sf
import torch


class VoiceBankDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        noisy_dir,
        clean_dir,
        scp_path,
        fs=16000,
        length_in_seconds=8,
        random_start_point=False,
        train=True,
        num_data_per_epoch=None,
    ):
        super().__init__()
        self.fs = fs
        self.L = int(length_in_seconds * fs)
        self.random_start_point = random_start_point
        self.train = train
        self.num_data_per_epoch = num_data_per_epoch
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        # 从 scp 读取音频 ID 列表
        with open(scp_path) as f:
            ids = [line.strip() for line in f if line.strip()]

        if not ids:
            raise RuntimeError(f"scp 文件 {scp_path} 中没有读到任何 ID")

        self.ids = ids
        print(f"VoiceBankDataset: scp={scp_path} → {len(self.ids)} 条 (fs={fs}, length={length_in_seconds}s)")

    def sample_data_per_epoch(self):
        """Trainer 在每个 epoch 开始时调用，用于从全集中随机采样子集。"""
        if self.train and self.num_data_per_epoch and self.num_data_per_epoch < len(self.ids):
            self.epoch_indices = random.sample(range(len(self.ids)), self.num_data_per_epoch)
        else:
            self.epoch_indices = None

    def __getitem__(self, idx):
        if hasattr(self, "epoch_indices") and self.epoch_indices is not None:
            idx = self.epoch_indices[idx]

        uid = self.ids[idx]
        noisy_path = f"{self.noisy_dir}/{uid}.wav"
        clean_path = f"{self.clean_dir}/{uid}.wav"

        if self.random_start_point:
            duration = sf.info(noisy_path).duration
            max_start = max(0, int(duration * self.fs) - self.L)
            start = random.randint(0, max_start) if max_start > 0 else 0
            noisy, _ = sf.read(noisy_path, dtype="float32", start=start, stop=start + self.L)
            clean, _ = sf.read(clean_path, dtype="float32", start=start, stop=start + self.L)
        else:
            noisy, _ = sf.read(noisy_path, dtype="float32")
            clean, _ = sf.read(clean_path, dtype="float32")

        # 统一长度：短补零，长截断
        if len(noisy) < self.L:
            noisy = np.pad(noisy, (0, self.L - len(noisy)))
        else:
            noisy = noisy[:self.L]

        if len(clean) < self.L:
            clean = np.pad(clean, (0, self.L - len(clean)))
        else:
            clean = clean[:self.L]

        return noisy, clean

    def __len__(self):
        if hasattr(self, "epoch_indices") and self.epoch_indices is not None:
            return len(self.epoch_indices)
        return len(self.ids)
