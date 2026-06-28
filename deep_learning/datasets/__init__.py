"""
数据集注册表 — 通过 cfg_train.yaml 中的 dataset.name 选择并实例化数据集。
"""

from .dns3_dataset import DNS3Dataset


DATASET_REGISTRY = {
    "dns3": DNS3Dataset,
}


def build_dataset(name: str, params: dict = None):
    """根据名称和参数字典实例化数据集。

    Args:
        name: 数据集名称，必须在 DATASET_REGISTRY 中
        params: 传递给数据集构造函数的参数字典

    Returns:
        torch.utils.data.Dataset 实例

    Raises:
        KeyError: 数据集名称不在注册表中
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"未知数据集 '{name}'，可用: {list(DATASET_REGISTRY.keys())}"
        )
    cls = DATASET_REGISTRY[name]
    if params is None:
        params = {}
    return cls(**params)
