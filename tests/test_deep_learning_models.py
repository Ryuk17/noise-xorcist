"""
测试 deep_learning 各模型的实例化和前向传播。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "deep_learning"))

import torch
from omegaconf import OmegaConf


class TestGTCRN:
    def test_forward_shape(self):
        from models import build_model
        cfg = OmegaConf.load(
            Path(__file__).parent.parent / "deep_learning" / "configs" / "cfg_train.yaml"
        )
        model = build_model("gtcrn", dict(cfg.model.params)).eval()
        x = torch.randn(2, 16000)
        with torch.inference_mode():
            y = model(x)
        assert y.shape == x.shape

    def test_causality(self):
        from models import build_model
        cfg = OmegaConf.load(
            Path(__file__).parent.parent / "deep_learning" / "configs" / "cfg_train.yaml"
        )
        model = build_model("gtcrn", dict(cfg.model.params)).eval()
        a = torch.randn(1, 16000)
        b = torch.randn(1, 16000)
        c = torch.randn(1, 16000)
        x1 = torch.cat([a, b], dim=1)
        x2 = torch.cat([a, c], dim=1)
        with torch.inference_mode():
            y1 = model(x1)
            y2 = model(x2)
        lookahead = 256 * 2
        diff = (y1[:, :16000 - lookahead] - y2[:, :16000 - lookahead]).abs().max()
        assert diff < 1e-3, f"因果性检查失败，差异={diff:.6f}"


class TestModelInstantiation:
    """CRN / DPCRN 模型需特定 STFT 特征维度输入，此处仅验证实例化。"""
    def test_crn_instantiates(self):
        from models import build_model
        model = build_model("crn", {"lstm_hidden_dim": 256, "n_lstm_layers": 2})
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_dpcrn_instantiates(self):
        from models import build_model
        model = build_model("dpcrn", {})
        assert sum(p.numel() for p in model.parameters()) > 0
