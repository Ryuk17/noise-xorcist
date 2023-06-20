import torch
import yaml
import argparse
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from models import *
from losses import *
from datasets import get_dataset, get_sampler
from models import get_model
from utils.utils import fix_seeds, setup_cudnn


def main(cfg):
    start = time.time()
    fix_seeds(123)
    setup_cudnn()

    device = cfg['DEVICE']
    device = torch.device(device)

    dataset = cfg['DATA']['DATASET']
    num_workers = cfg['DATA']['NUM_WORKERS']

    epoch = cfg['TRAIN']['EPOCHS']
    batch_size = cfg['TRAIN']['batch_size']
    eval_interval = cfg['TRAIN']['EVAL_INTERVAL']
    lr = cfg['TRAIN']['LR']
    optimizer = cfg['TRAIN']['OPTIMIER']

    amp = cfg['TRAIN']['AMP']
    ddp = cfg['TRAIN']['DDP']

    pretrained = cfg['MODEL']['PRETRAINED']
    model = get_model(cfg)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))

    train_dataset = get_dataset(cfg['DATA'], 'train')
    val_dataset = get_dataset(cfg['DATA'], 'val')

    train_sampler, val_sampler = get_sampler(ddp, train_dataset, val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    end = time.gmtime(time.time() - start)
    total_time = time.strftime("%H:%M:%S", end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/finetune.yaml')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)