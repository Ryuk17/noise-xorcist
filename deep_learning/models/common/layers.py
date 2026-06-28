"""
CRN / DPCRN / GTCRN 系列模型共享的基础层和编解码器。
纯 torch 依赖，不依赖 df/libdf 等 DeepFilterNet 专用包。
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 原子层
# ---------------------------------------------------------------------------


class Chomp_T(nn.Module):
    """因果卷积的时间维度裁剪，去掉未来的 padding 部分。"""
    def __init__(self, chomp_t):
        super().__init__()
        self.chomp_t = chomp_t

    def forward(self, x):
        return x[:, :, 0:-self.chomp_t, :]


def complex_ratio_mask(mask, spec):
    """复数比值掩膜: enhanced = spec * mask (complex multiplication)

    Args:
        mask: (B, 2, T, F) — 实部 + 虚部
        spec: (B, 2, T, F) — 实部 + 虚部
    Returns:
        (B, 2, T, F) — 增强后的实部 + 虚部
    """
    enh_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
    enh_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
    return torch.stack([enh_real, enh_imag], dim=1)


# ---------------------------------------------------------------------------
# 组合结构
# ---------------------------------------------------------------------------


class CausalConvEncoder(nn.Module):
    """5 阶段因果卷积编码器，时间维度因果、频率逐阶段减半。

    Args:
        channels: 各阶段通道数 [in_ch, c1, c2, c3, c4, c5]，长度 6
        activation: 激活函数类，如 nn.ELU, nn.PReLU
    """

    def __init__(self, channels, activation=nn.ELU):
        super().__init__()
        pad = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        modules = []
        for i in range(len(channels) - 1):
            modules.append(nn.Sequential(
                pad,
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=(2, 3), stride=(1, 2)),
                nn.BatchNorm2d(channels[i + 1]),
                activation(),
            ))
        self.en_module = nn.ModuleList(modules)

    def forward(self, x):
        x_list = []
        for layer in self.en_module:
            x = layer(x)
            x_list.append(x)
        return x, x_list


class CausalConvDecoder(nn.Module):
    """5 阶段因果转置卷积解码器，带 concat skip connection。

    各阶段顺序: ConvTranspose2d → [freq_pad] → Chomp_T(1) → [BN] → activation

    Args:
        channels: [latent_ch, c4, c3, c2, c1, out_ch]，长度 6
                  每阶段实际输入 = channels[i] * 2（concat encoder skip）
        activation: 中间层激活函数类
        final_activation: 最终层激活类（None 表示无激活）
        use_bn_last: 最后一层是否加 BatchNorm
        freq_pad_stage: 需要频率 padding 的阶段（0-indexed，crn/dpcrn 为第 4 阶段即 index 3）
    """

    def __init__(self, channels, activation=nn.ELU, final_activation=nn.Softplus,
                 use_bn_last=True, freq_pad_stage=3):
        super().__init__()
        n_stages = len(channels) - 1
        freq_pad = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        modules = []
        for i in range(n_stages):
            is_last = (i == n_stages - 1)
            layers = [
                nn.ConvTranspose2d(channels[i] * 2, channels[i + 1],
                                   kernel_size=(2, 3), stride=(1, 2)),
            ]
            if i == freq_pad_stage:
                layers.append(freq_pad)

            layers.append(Chomp_T(1))

            if not is_last or use_bn_last:
                layers.append(nn.BatchNorm2d(channels[i + 1]))

            act = final_activation if is_last and final_activation is not None else activation
            layers.append(act())

            modules.append(nn.Sequential(*layers))
        self.de_module = nn.ModuleList(modules)

    def forward(self, x, x_list):
        for i, layer in enumerate(self.de_module):
            x = torch.cat((x, x_list[-(i + 1)]), dim=1)
            x = layer(x)
        return x.squeeze(dim=1)
