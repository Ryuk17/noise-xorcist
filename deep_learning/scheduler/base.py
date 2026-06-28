"""
学习率调度器基类。
"""
from abc import abstractmethod
from torch.optim import lr_scheduler


class BaseLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    @abstractmethod
    def get_lr(self):
        raise NotImplementedError

    @abstractmethod
    def reinitialize(self, **kwargs) -> None:
        raise NotImplementedError
