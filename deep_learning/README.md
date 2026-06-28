<!--
 * @Author: Ryuk
 * @Date: 2026-02-17 14:59:35
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-06-28
 * @Description: Deep learning speech enhancement training framework.
-->

## 目录结构

| 目录 | 说明 |
|------|------|
| `configs/` | YAML 配置文件（训练 / 推理） |
| `models/` | 模型定义，通过注册表按名称实例化 |
| `models/common/` | 因果编解码器等通用复用组件 |
| `models/deepfilternet/` | DeepFilterNet 系列模型 |
| `losses/` | 损失函数，通过注册表按名称实例化 |
| `datasets/` | 数据集，通过注册表按名称实例化 |
| `scheduler/` | 学习率调度器，通过注册表按名称实例化 |
| `utils/` | DDP 分布式训练等工具 |

顶层脚本：`train.py`（训练）、`infer.py`（推理）、`evaluate.py`（评估）、`dataloader.py`（DataLoader 测试）。

## 配置驱动设计

模型、损失函数、数据集、调度器均通过 `configs/cfg_train.yaml` 配置，无需修改 `train.py` 代码。

### cfg_train.yaml 关键配置

```yaml
model:
  name: gtcrn               # 模型名称，对应 MODEL_REGISTRY
  params:                   # 模型构造参数
    n_fft: 512
    hop_len: 256
    win_len: 512

loss:
  name: hybrid              # 损失函数名称，对应 LOSS_REGISTRY
  params:                   # 损失函数构造参数
    n_fft: 512
    ...

train_dataset:
  name: dns3                # 数据集名称，对应 DATASET_REGISTRY
  params:                   # 数据集构造参数
    length_in_seconds: 10
    ...

validation_dataset:
  name: dns3
  params: ...

scheduler:
  name: warmup_cosine       # 调度器名称，对应 SCHEDULER_REGISTRY
  params:                   # 调度器构造参数
    warmup_steps: 25000
    ...
  update_interval: step     # step 或 epoch
```

### 可用组件

| 组件 | 注册表 | 可用名称 |
|------|--------|----------|
| 模型 | `MODEL_REGISTRY` | `gtcrn`, `crn`, `gcrn`, `gccrn`, `dpcrn`, `nsnet`, `df1`, `df2`, `df3` |
| 损失函数 | `LOSS_REGISTRY` | `hybrid`, `stft`, `multi_stft`, `compressed_mse`, `weighted_sd`, `neg_snr`, `gain_neg_snr`, `sisnr` |
| 数据集 | `DATASET_REGISTRY` | `dns3` |
| 调度器 | `SCHEDULER_REGISTRY` | `warmup_cosine`, `step`, `multistep`, `cosine`, `plateau` |

## 使用流程

1. **准备数据集**：在 `datasets/` 中创建数据集类，并在 `datasets/__init__.py` 注册
2. **定义模型**：在 `models/` 中创建模型文件，并在 `models/__init__.py` 注册
3. **选择/创建损失函数**：在 `losses/` 中定义，并在 `losses/__init__.py` 注册
4. **配置训练**：修改 `configs/cfg_train.yaml` 中的 name + params
5. **运行训练**：
   ```bash
   python train.py                              # 单卡
   python train.py -D 1                         # 指定 GPU
   python train.py -C configs/cfg_train.yaml -D 0,1,2,3  # 多卡 DDP
   ```
6. **推理**：在 `configs/cfg_infer.yaml` 中指定 checkpoint 路径，运行 `python infer.py`
7. **评估**：运行 `python evaluate.py`

## 添加新组件

只需两步，无需修改 train.py：

1. 创建实现文件
2. 在对应的 `__init__.py` 注册表中添加一行

例如添加新数据集：
```python
# datasets/my_dataset.py — 实现 MyDataset 类
# datasets/__init__.py
from .my_dataset import MyDataset
DATASET_REGISTRY["my_dataset"] = MyDataset
```

然后在 `cfg_train.yaml` 中：
```yaml
train_dataset:
  name: my_dataset
  params: ...
```

## 注意事项

1. 代码面向 Linux 系统，Windows 可能遇到路径兼容和 pesq 安装问题
2. DeepFilterNet 系列模型 (`df1`/`df2`/`df3`) 需要额外安装 `libdf` 和 `df` 包，参考 [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
3. 如对本项目有帮助，欢迎 star

## 致谢

本代码模板大量参考了优秀的 [SEtrain](https://github.com/Xiaobin-Rong/SEtrain/tree/plus) 仓库。
