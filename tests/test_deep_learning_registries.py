"""
测试 deep_learning 四大注册表：模型 / 损失函数 / 数据集 / 调度器。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "deep_learning"))

import torch
import torch.nn as nn
from omegaconf import OmegaConf

_CFG = None

def _config():
    global _CFG
    if _CFG is None:
        _CFG = OmegaConf.load(
            Path(__file__).parent.parent / "deep_learning" / "configs" / "cfg_train.yaml"
        )
    return _CFG


class TestModelRegistry:
    def test_all_names_exist(self):
        from models import MODEL_REGISTRY
        expected = {"gtcrn", "crn", "gcrn", "gccrn", "dpcrn", "nsnet", "df1", "df2", "df3"}
        assert set(MODEL_REGISTRY.keys()) == expected

    def test_build_gtcrn(self):
        from models import build_model
        m = build_model("gtcrn", dict(_config().model.params))
        assert isinstance(m, nn.Module)

    def test_build_crn(self):
        from models import build_model
        m = build_model("crn", {"lstm_hidden_dim": 256, "n_lstm_layers": 2})
        assert isinstance(m, nn.Module)

    def test_build_dpcrn(self):
        from models import build_model
        m = build_model("dpcrn", {})
        assert isinstance(m, nn.Module)

    def test_build_nsnet(self):
        from models import build_model
        m = build_model("nsnet", {"input_dim": 256, "n_gru_layers": 3, "gru_dropout": 0.1})
        assert isinstance(m, nn.Module)

    def test_unknown_model_raises(self):
        from models import build_model
        try:
            build_model("nonexistent", {})
            assert False, "应该抛出 KeyError"
        except KeyError:
            pass


class TestLossRegistry:
    def test_all_names_exist(self):
        from losses import LOSS_REGISTRY
        expected = {
            "hybrid", "stft", "multi_stft", "compressed_mse",
            "weighted_sd", "neg_snr", "gain_neg_snr", "sisnr",
        }
        assert set(LOSS_REGISTRY.keys()) == expected

    def test_build_hybrid(self):
        from losses import build_loss
        l = build_loss("hybrid", dict(_config().loss.params))
        assert isinstance(l, nn.Module)

    def test_build_sisnr(self):
        from losses import build_loss
        l = build_loss("sisnr", {})
        assert isinstance(l, nn.Module)

    def test_unknown_loss_raises(self):
        from losses import build_loss
        try:
            build_loss("nonexistent", {})
            assert False, "应该抛出 KeyError"
        except KeyError:
            pass


class TestDatasetRegistry:
    def test_all_names_exist(self):
        from datasets import DATASET_REGISTRY
        assert set(DATASET_REGISTRY.keys()) == {"dns3"}

    def test_dns3_class(self):
        from datasets import DATASET_REGISTRY, DNS3Dataset
        assert DATASET_REGISTRY["dns3"] is DNS3Dataset


class TestSchedulerRegistry:
    def test_all_names_exist(self):
        from scheduler import SCHEDULER_REGISTRY
        expected = {"warmup_cosine", "step", "multistep", "cosine", "plateau"}
        assert set(SCHEDULER_REGISTRY.keys()) == expected

    def test_build_warmup_cosine(self):
        from scheduler import build_scheduler
        opt = torch.optim.Adam(nn.Linear(10, 10).parameters(), lr=0.001)
        s = build_scheduler("warmup_cosine", opt,
                            {"warmup_steps": 100, "decay_until_step": 1000,
                             "max_lr": 1e-3, "min_lr": 1e-6})
        assert s is not None

    def test_build_step(self):
        from scheduler import build_scheduler
        opt = torch.optim.Adam(nn.Linear(10, 10).parameters(), lr=0.001)
        s = build_scheduler("step", opt, {"step_size": 30, "gamma": 0.1})
        assert isinstance(s, torch.optim.lr_scheduler.StepLR)

    def test_unknown_scheduler_raises(self):
        from scheduler import build_scheduler
        opt = torch.optim.Adam(nn.Linear(10, 10).parameters(), lr=0.001)
        try:
            build_scheduler("nonexistent", opt, {})
            assert False, "应该抛出 KeyError"
        except KeyError:
            pass
