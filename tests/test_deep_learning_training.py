"""
训练流程冒烟测试 — 验证各模型前向+反向传播正常，loss 在真实数据上下降。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "deep_learning"))

import torch
from omegaconf import OmegaConf

from models import build_model, MODEL_REGISTRY
from losses import build_loss
from datasets import build_dataset

_CFG = None


def _config():
    global _CFG
    if _CFG is None:
        _CFG = OmegaConf.load(
            Path(__file__).parent.parent / "deep_learning" / "configs" / "cfg_train.yaml"
        )
    return _CFG


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stft(x, n_fft=512, hop_len=256, win_len=512):
    """返回 (real, imag) 拼接: (B, 2, T, F)。"""
    window = torch.hann_window(win_len)
    X = torch.stft(x, n_fft, hop_len, win_len, window, return_complex=True)
    return torch.stack([X.real, X.imag], dim=1)


def _fake_waveform(batch=2, seconds=2, sr=16000):
    return torch.randn(batch, int(seconds * sr))


def _grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _train_steps(model, loss_func, dataloader, device, steps=5):
    """执行几步训练，返回 loss 列表。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    it = iter(dataloader)
    for _ in range(steps):
        try:
            noisy, clean = next(it)
        except StopIteration:
            it = iter(dataloader)
            noisy, clean = next(it)
        noisy = noisy.to(device)
        clean = clean.to(device)
        enhanced = model(noisy)
        loss = loss_func(enhanced, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------------------
# 1. 前向+反向冒烟测试 (合成数据)
# ---------------------------------------------------------------------------

class TestForwardBackwardSmoke:
    """所有已注册模型至少能完成一次前向+反向传播。"""

    @staticmethod
    def _run(name, params, make_input):
        cfg = _config()
        model = build_model(name, params)
        loss_func = build_loss(cfg.loss.name, dict(cfg.loss.params))
        noisy, clean = make_input()
        enhanced = model(noisy)
        loss = loss_func(enhanced, clean)
        loss.backward()
        grad = _grad_norm(model)
        assert grad > 0, f"{name}: 梯度为零"
        assert torch.isfinite(loss), f"{name}: loss 非有限 ({loss.item()})"

    def test_gtcrn(self):
        self._run("gtcrn",
                  dict(n_fft=512, hop_len=256, win_len=512),
                  lambda: (_fake_waveform(), _fake_waveform()))

    def test_crn(self):
        n, h, w = 512, 256, 512
        def make():
            x = _fake_waveform()
            X = _stft(x, n, h, w)
            S = X[:, 0]  # (B, T=63, F=257) — just use magnitude for now
            # Wait, CRN expects (B, T, F) with channel=1 internally added
            # Actually from the code: x = x.unsqueeze(dim=1) so input is (B, T, F)
            # The encoder does channels=[1,16,...] so it expects (B, 1, T, F)
            # Wait, unsqueeze(1) adds a dim at pos 1, so input is (B, T, F) -> (B, 1, T, F)
            return S, S  # (B, T, F)
        self._run("crn", {"lstm_hidden_dim": 256, "n_lstm_layers": 2}, make)

    def test_dpcrn(self):
        n, h, w = 512, 256, 512
        def make():
            x = _fake_waveform()
            X = _stft(x, n, h, w)  # (B, 2, T, F)
            return X, X
        self._run("dpcrn", {}, make)

    def test_gcrn(self):
        n, h, w = 512, 256, 512
        def make():
            x = _fake_waveform()
            X = _stft(x, n, h, w)  # (B, 2, T, F)
            return X, X
        self._run("gcrn", {}, make)

    def test_gccrn(self):
        # GCCRN expects (B, time, channel=4, bin=161)
        # Construct: stack two 2-channel STFTs or just use 4-channel fake
        n = 256
        h = n // 2
        w = n
        def make():
            x = _fake_waveform(sr=16000)
            X = _stft(x, n, h, w)          # (B, 2, T, F)
            X4 = torch.cat([X, X], dim=1)   # (B, 4, T, F)
            return X4.transpose(1, 2), X4.transpose(1, 2)  # (B, T, 4, F)
        self._run("gccrn", {}, make)

    def test_nsnet(self):
        n = 256
        h = n // 2
        w = n
        def make():
            x = _fake_waveform(sr=16000)
            X = _stft(x, n, h, w)          # (B, 2, T, F)
            mag = torch.sqrt(X[:, 0]**2 + X[:, 1]**2 + 1e-12)  # (B, T, F=129)
            out = mag.permute(0, 2, 1)      # (B, F, T) — what NSNet expects
            return out, mag
        self._run("nsnet", {"input_dim": 129, "n_gru_layers": 3, "gru_dropout": 0.1}, make)


# ---------------------------------------------------------------------------
# 2. 真实 VoiceBank 数据训练测试
# ---------------------------------------------------------------------------

class TestTrainingOnVoiceBank:
    """用 VoiceBank 验证集数据训练几步，确认 loss 下降。"""

    @classmethod
    def setup_class(cls):
        cfg = _config()
        params = dict(cfg.validation_dataset.params)
        params["length_in_seconds"] = 2
        cls.dataset = build_dataset("voicebank", params)
        cls.loss_func = build_loss(cfg.loss.name, dict(cfg.loss.params))

    def test_gtcrn_loss_decreases(self):
        model = build_model("gtcrn", dict(n_fft=512, hop_len=256, win_len=512))
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=2, shuffle=True)
        losses = _train_steps(model, self.loss_func, loader, torch.device("cpu"), steps=8)
        assert losses[-1] < losses[0], (
            f"GTCRN loss 未下降: {losses}"
        )

    def test_gtcrn_overfit_one_batch(self):
        """单 batch 过拟合 — loss 应显著下降。"""
        model = build_model("gtcrn", dict(n_fft=512, hop_len=256, win_len=512))
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=2, shuffle=True)
        noisy, clean = next(iter(loader))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_func = self.loss_func

        losses = []
        for _ in range(30):
            enhanced = model(noisy)
            loss = loss_func(enhanced, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.5, (
            f"Loss 下降不足: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. 配置有效性
# ---------------------------------------------------------------------------

def test_cfg_train_yaml_valid():
    """cfg_train.yaml 中的所有组件名都在对应注册表中。"""
    cfg = _config()
    assert cfg.model.name in MODEL_REGISTRY, f"model '{cfg.model.name}' 未注册"
    from losses import LOSS_REGISTRY
    assert cfg.loss.name in LOSS_REGISTRY, f"loss '{cfg.loss.name}' 未注册"
    from datasets import DATASET_REGISTRY
    assert cfg.train_dataset.name in DATASET_REGISTRY, f"train_dataset '{cfg.train_dataset.name}' 未注册"
    assert cfg.validation_dataset.name in DATASET_REGISTRY, f"validation_dataset '{cfg.validation_dataset.name}' 未注册"
    from scheduler import SCHEDULER_REGISTRY
    assert cfg.scheduler.name in SCHEDULER_REGISTRY, f"scheduler '{cfg.scheduler.name}' 未注册"
