"""
DataLoader 测试脚本 — 验证数据集和 DataLoader 的组合使用。
具体数据集实现请见 datasets/ 目录。
"""
from torch.utils import data


if __name__ == '__main__':
    from tqdm import tqdm
    from omegaconf import OmegaConf
    from datasets import build_dataset

    config = OmegaConf.load('configs/cfg_train.yaml')

    train_dataset = build_dataset(config['train_dataset']['name'], config['train_dataset']['params'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    train_dataloader.dataset.sample_data_per_epoch()

    validation_dataset = build_dataset(config['validation_dataset']['name'], config['validation_dataset']['params'])
    validation_dataloader = data.DataLoader(validation_dataset, **config['validation_dataloader'])

    print(len(train_dataloader), len(validation_dataloader))

    for noisy, clean in tqdm(train_dataloader):
        print(noisy.shape, clean.shape)
        break

    for noisy, clean in tqdm(validation_dataloader):
        print(noisy.shape, clean.shape)
        break
