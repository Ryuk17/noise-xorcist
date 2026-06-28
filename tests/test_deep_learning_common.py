"""
测试 deep_learning models/common 中的通用复用组件。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "deep_learning"))

import torch
import torch.nn as nn
from models.common.layers import (
    Chomp_T,
    complex_ratio_mask,
    CausalConvEncoder,
    CausalConvDecoder,
)


class TestChompT:
    def test_chomp_removes_time(self):
        layer = Chomp_T(chomp_t=2)
        x = torch.randn(1, 16, 10, 32)
        y = layer(x)
        assert y.shape == (1, 16, 8, 32)

    def test_chomp_one(self):
        layer = Chomp_T(chomp_t=1)
        x = torch.randn(2, 8, 10, 16)
        y = layer(x)
        assert y.shape == (2, 8, 9, 16)


class TestComplexRatioMask:
    def test_shape(self):
        mask = torch.randn(2, 2, 10, 32)
        spec = torch.randn(2, 2, 10, 32)
        out = complex_ratio_mask(mask, spec)
        assert out.shape == (2, 2, 10, 32)

    def test_identity_mask(self):
        spec = torch.randn(1, 2, 4, 8)
        mask = torch.zeros(1, 2, 4, 8)
        mask[:, 0] = 1.0
        out = complex_ratio_mask(mask, spec)
        assert torch.allclose(out, spec, atol=1e-6)


class TestCausalConvEncoder:
    def test_output_shape(self):
        """5 阶段 stride=2 频率减半: 256→127→63→31→15→7"""
        enc = CausalConvEncoder(
            channels=[1, 16, 32, 64, 128, 256], activation=nn.ELU,
        )
        x = torch.randn(2, 1, 100, 256)
        out, skip_list = enc(x)
        assert out.shape == (2, 256, 100, 7)
        assert len(skip_list) == 5

    def test_dpcrn_channels(self):
        enc = CausalConvEncoder(
            channels=[2, 32, 32, 32, 64, 128], activation=nn.PReLU,
        )
        x = torch.randn(1, 2, 50, 128)
        out, _ = enc(x)
        assert out.shape == (1, 128, 50, 3)


class TestCausalConvDecoder:
    def test_instantiates(self):
        """解码器实例化验证（前向正确性由模型集成测试覆盖）"""
        dec = CausalConvDecoder(
            channels=[256, 128, 64, 32, 16, 1],
            activation=nn.ELU, final_activation=nn.Softplus, use_bn_last=True,
        )
        assert isinstance(dec, nn.Module)
