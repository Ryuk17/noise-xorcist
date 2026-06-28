"""
模型注册表 — 通过 cfg_train.yaml 中的 model.name 选择并实例化模型。
注册表的值可以是类（直接透传 params）或工厂函数（特殊初始化逻辑）。
"""

from .gtcrn import GTCRN
from .crn import build_crn
from .gcrn import GcrnModel
from .gccrn import GCCRN
from .dpcrn import DpcrnModel
from .nsnet import NSNetModel


# DeepFilterNet 各版本 — 延迟导入，避免 libdf 未安装时阻塞其他模型
def _build_df1(**params):
    from .deepfilternet import init_df1
    return init_df1(**params)


def _build_df2(**params):
    from .deepfilternet import init_df2
    return init_df2(**params)


def _build_df3(**params):
    from .deepfilternet import init_df3
    return init_df3(**params)


MODEL_REGISTRY = {
    "gtcrn": GTCRN,
    "crn": build_crn,
    "gcrn": GcrnModel,
    "gccrn": GCCRN,
    "dpcrn": DpcrnModel,
    "nsnet": NSNetModel,
    "df1": _build_df1,
    "df2": _build_df2,
    "df3": _build_df3,
}


def build_model(name: str, params: dict = None):
    """根据名称和参数字典实例化模型。

    Args:
        name: 模型名称，必须在 MODEL_REGISTRY 中
        params: 传递给模型构造函数/工厂函数的参数字典

    Returns:
        nn.Module 实例

    Raises:
        KeyError: 模型名称不在注册表中
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"未知模型 '{name}'，可用模型: {list(MODEL_REGISTRY.keys())}"
        )
    factory = MODEL_REGISTRY[name]
    if params is None:
        params = {}
    return factory(**params)
