"""
调度器注册表 — 通过 cfg_train.yaml 中的 scheduler.name 选择并实例化调度器。
"""

import torch.optim.lr_scheduler as _lrs

from .warmup_cosine import LinearWarmupCosineAnnealingLR


SCHEDULER_REGISTRY = {
    "warmup_cosine": LinearWarmupCosineAnnealingLR,
    "step": _lrs.StepLR,
    "multistep": _lrs.MultiStepLR,
    "cosine": _lrs.CosineAnnealingLR,
    "plateau": _lrs.ReduceLROnPlateau,
}


def build_scheduler(name: str, optimizer, params: dict = None):
    """根据名称和参数字典实例化调度器。

    Args:
        name: 调度器名称，必须在 SCHEDULER_REGISTRY 中
        optimizer: torch.optim.Optimizer 实例
        params: 传递给调度器构造函数的参数字典

    Returns:
        调度器实例

    Raises:
        KeyError: 调度器名称不在注册表中
    """
    if name not in SCHEDULER_REGISTRY:
        raise KeyError(
            f"未知调度器 '{name}'，可用: {list(SCHEDULER_REGISTRY.keys())}"
        )
    cls = SCHEDULER_REGISTRY[name]
    if params is None:
        params = {}
    return cls(optimizer, **params)
