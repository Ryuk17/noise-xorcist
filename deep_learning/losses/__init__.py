"""
损失函数注册表 — 通过 cfg_train.yaml 中的 loss.name 选择并实例化损失函数。
"""

from .mse_loss import (
    WeightedSpeechDistortionLoss,
    ComplexCompressedMSELoss,
    STFTLoss,
    MultiResolutionSTFTLoss,
)

from .snr_loss import (
    NegativeSNRLoss,
    GainMaskBasedNegativeSNRLoss,
    SISNRLoss,
)

from .hybrid_loss import (
    HybridLoss,
)


LOSS_REGISTRY = {
    "hybrid": HybridLoss,
    "stft": STFTLoss,
    "multi_stft": MultiResolutionSTFTLoss,
    "compressed_mse": ComplexCompressedMSELoss,
    "weighted_sd": WeightedSpeechDistortionLoss,
    "neg_snr": NegativeSNRLoss,
    "gain_neg_snr": GainMaskBasedNegativeSNRLoss,
    "sisnr": SISNRLoss,
}


def build_loss(name: str, params: dict = None):
    """根据名称和参数字典实例化损失函数。

    Args:
        name: 损失函数名称，必须在 LOSS_REGISTRY 中
        params: 传递给损失函数构造函数的参数字典

    Returns:
        nn.Module 实例

    Raises:
        KeyError: 损失函数名称不在注册表中
    """
    if name not in LOSS_REGISTRY:
        raise KeyError(
            f"未知损失函数 '{name}'，可用: {list(LOSS_REGISTRY.keys())}"
        )
    cls = LOSS_REGISTRY[name]
    if params is None:
        params = {}
    return cls(**params)
