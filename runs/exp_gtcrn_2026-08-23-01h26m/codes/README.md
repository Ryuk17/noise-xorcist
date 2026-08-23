<!--
 * @Author: Ryuk
 * @Date: 2026-02-17 14:59:35
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-06-28
 * @Description: Deep learning speech enhancement training framework.
-->

## Directory Structure

| Directory | Description |
|-----------|-------------|
| `configs/` | YAML config files (training / inference) |
| `models/` | Model definitions, instantiated by name via registry |
| `models/common/` | Reusable components such as causal encoder/decoder |
| `models/deepfilternet/` | DeepFilterNet model family |
| `losses/` | Loss functions, instantiated by name via registry |
| `datasets/` | Datasets, instantiated by name via registry |
| `scheduler/` | Learning rate schedulers, instantiated by name via registry |
| `utils/` | DDP distributed training utilities |

Top-level scripts: `train.py` (training), `infer.py` (inference), `evaluate.py` (evaluation), `dataloader.py` (DataLoader smoke test).

## Config-Driven Design

Models, loss functions, datasets, and schedulers are all configured through `configs/cfg_train.yaml` — no changes to `train.py` are needed.

### Key cfg_train.yaml Sections

```yaml
model:
  name: gtcrn               # model name, maps to MODEL_REGISTRY
  params:                   # model constructor arguments
    n_fft: 512
    hop_len: 256
    win_len: 512

loss:
  name: hybrid              # loss name, maps to LOSS_REGISTRY
  params:                   # loss constructor arguments
    n_fft: 512
    ...

train_dataset:
  name: dns3                # dataset name, maps to DATASET_REGISTRY
  params:                   # dataset constructor arguments
    length_in_seconds: 10
    ...

validation_dataset:
  name: dns3
  params: ...

scheduler:
  name: warmup_cosine       # scheduler name, maps to SCHEDULER_REGISTRY
  params:                   # scheduler constructor arguments
    warmup_steps: 25000
    ...
  update_interval: step     # "step" or "epoch"
```

### Available Components

| Component | Registry | Available Names |
|-----------|----------|-----------------|
| Model | `MODEL_REGISTRY` | `gtcrn`, `crn`, `gcrn`, `gccrn`, `dpcrn`, `nsnet`, `df1`, `df2`, `df3` |
| Loss | `LOSS_REGISTRY` | `hybrid`, `stft`, `multi_stft`, `compressed_mse`, `weighted_sd`, `neg_snr`, `gain_neg_snr`, `sisnr` |
| Dataset | `DATASET_REGISTRY` | `dns3`, `voicebank` |
| Scheduler | `SCHEDULER_REGISTRY` | `warmup_cosine`, `step`, `multistep`, `cosine`, `plateau` |

## Usage

1. **Prepare dataset**: Create a dataset class in `datasets/` and register it in `datasets/__init__.py`
2. **Define model**: Create a model file in `models/` and register it in `models/__init__.py`
3. **Select/create loss**: Define it in `losses/` and register it in `losses/__init__.py`
4. **Configure training**: Set name + params in `configs/cfg_train.yaml`
5. **Run training**:
   ```bash
   python train.py                              # single GPU
   python train.py -D 1                         # specific GPU
   python train.py -C configs/cfg_train.yaml -D 0,1,2,3  # multi-GPU DDP
   ```
6. **Inference**: Specify the checkpoint path in `configs/cfg_infer.yaml`, then run `python infer.py`
7. **Evaluation**: Run `python evaluate.py`

## Adding a New Component

Only two steps, no changes to `train.py`:

1. Create the implementation file
2. Add one line to the corresponding `__init__.py` registry

Example for adding a new dataset:
```python
# datasets/my_dataset.py — implement MyDataset class
# datasets/__init__.py
from .my_dataset import MyDataset
DATASET_REGISTRY["my_dataset"] = MyDataset
```

Then in `cfg_train.yaml`:
```yaml
train_dataset:
  name: my_dataset
  params: ...
```

## Notes

1. This code targets Linux. Windows may encounter path compatibility and `pesq` installation issues.
2. DeepFilterNet models (`df1`/`df2`/`df3`) require separate installation of `libdf` and `df` packages — see [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet).
3. If you find this project useful, a star is appreciated.

## Acknowledgements

This code template draws heavily from the excellent [SEtrain](https://github.com/Xiaobin-Rong/SEtrain/tree/plus) repository.
