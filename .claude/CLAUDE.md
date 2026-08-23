# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

noise-xorcist is a unified single-channel speech enhancement toolbox with two tracks:
- **Signal processing**: Traditional DSP-based noise estimation and spectral gain estimation (no training required)
- **Deep learning**: PyTorch-based training/inference pipeline with config-driven model/loss/dataset/scheduler selection

## Commands

### Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_noise_estimation.py     # traditional DSP noise estimators
python -m pytest tests/test_spectral_gain_estimation.py  # traditional DSP gain estimators
python -m pytest tests/test_deep_learning_models.py  # DL model instantiation & forward pass
python -m pytest tests/test_deep_learning_common.py  # common layers (causal conv, complex ratio mask)
python -m pytest tests/test_deep_learning_registries.py  # registry builder functions

# Run a single test
python -m pytest tests/test_deep_learning_models.py::TestGTCRN::test_forward_shape -v

# Run with coverage
python -m pytest tests/ --cov=./
```

### Training & Inference (Deep Learning)
```bash
# Train (single GPU)
cd deep_learning && python train.py

# Train with config and multi-GPU
python train.py -C configs/cfg_train.yaml -D 0,1,2,3

# Train on specific GPU
python train.py -D 1

# Inference
python infer.py -C configs/cfg_infer.yaml -D 0

# Evaluate metrics (直接传 scp, 或传 --enh_dir 从该目录读 inf.scp/ref.scp)
python ../evaluation/evaluate.py --metric intrusive --inf_scp <enh_dir>/inf.scp --ref_scp <enh_dir>/ref.scp
python ../evaluation/evaluate.py --metric dnsmos --inf_scp <enh_dir>/inf.scp [--device cpu]

# Test DataLoader
python dataloader.py
```

### Signal Processing (Traditional DSP)
```bash
# Run example (MCRA noise estimation + spectral subtraction)
python examples/mcra_spectral_substraction.py

# Test individual estimators
python tests/test_noise_estimation.py
python tests/test_spectral_gain_estimation.py
```

### Install
```bash
pip install -r requirements.txt
```

## Project Structure

```
noise-xorcist/
├── signal_processing/             # Traditional DSP methods (no GPU needed)
│   ├── base/
│   │   ├── denoiser.py            # BaseNoiseEstimator, BaseSpectralGainEstimator
│   ├── noise_estimation/          # 8 noise PSD estimators
│   │   ├── ms.py, mcra.py, mcra2.py, imcra.py
│   │   ├── csmt.py, wsa.py, cfr.py, spp.py
│   ├── spectral_gain_estimation/  # 10 gain estimators
│       ├── spectral_subtraction.py, mmse.py, logmmse.py, logmmse_spu.py
│       ├── stsa_mis.py, stsa_wcosh.py, stsa_weuclid.py, stsa_wlr.py
│       ├── wiener.py, omlsa.py
│
├── deep_learning/                 # PyTorch training/inference pipeline
│   ├── train.py                   # Entry point: Trainer class, multi-GPU support
│   ├── infer.py                   # Inference: loads checkpoint, processes wavs
│   ├── dataloader.py              # DataLoader smoke test script
│   ├── configs/
│   │   ├── cfg_train.yaml         # Training config (model/loss/dataset/scheduler/optimizer)
│   │   └── cfg_infer.yaml         # Inference config (test dirs, checkpoint path)
│   ├── models/
│   │   ├── __init__.py            # MODEL_REGISTRY — build_model(name, params)
│   │   ├── common/layers.py       # Shared: CausalConvEncoder/Decoder, complex_ratio_mask, Chomp_T
│   │   ├── deepfilternet/         # DeepFilterNet df1/df2/df3 (needs libdf)
│   │   ├── gtcrn.py, crn.py, gcrn.py, gccrn.py, dpcrn.py, nsnet.py
│   ├── losses/
│   │   ├── __init__.py            # LOSS_REGISTRY — build_loss(name, params)
│   │   ├── base.py                # BaseSELoss
│   │   ├── hybrid_loss.py         # HybridLoss (STFT magnitude + complex + SISNR)
│   │   ├── mse_loss.py            # WeightedSpeechDistortion, CompressedMSE, STFT, MultiResolutionSTFT
│   │   ├── snr_loss.py            # NegativeSNR, GainMaskNegativeSNR, SISNR
│   │   ├── phase_loss.py          # Phase-aware loss (available but not registered)
│   ├── datasets/
│   │   ├── __init__.py            # DATASET_REGISTRY — build_dataset(name, params)
│   │   └── dns3_dataset.py        # DNS3 challenge dataset
│   ├── scheduler/
│   │   ├── __init__.py            # SCHEDULER_REGISTRY — build_scheduler(name, optimizer, params)
│   │   ├── warmup_cosine.py       # LinearWarmupCosineAnnealingLR
│   │   └── base.py                # BaseScheduler
│   └── utils/
│       └── distributed_utils.py   # DDP helpers (init, reduce_value, cleanup)
│
├── evaluation/                    # Speech quality metrics
│   ├── calculate_intrusive_se_metrics.py   # SDR, SISNR, PESQ, ESTOI
│   ├── calculate_nonintrusive_dnsmos.py    # DNSMOS (OVRL, SIG, BAK, P808_MOS)
│   └── evaluate.py                # Runner script wrapping intrusive/non-intrusive metric scripts
│
├── noisyspeech_synthesizer/       # Noisy speech data generation
│   ├── prepare_custom_datasets/
│   │   ├── augmentations.py       # SpecAugment, MixAugment, VolAugment, ClipAugment, BreakAugment, HowlingAugment
│   │   ├── audiolib.py            # Audio I/O and mixing utilities
│   │   └── noisyspeech_synthesizer_multiprocessing.py  # Multi-process dataset generation
│   ├── prepare_DNS_datasets/      # DNS challenge-specific dataset preparation
│   └── utils.py
│
├── examples/
│   └── mcra_spectral_substraction.py   # End-to-end example comboing MCRA + spectral subtraction
│
├── tests/                         # pytest-based tests
│   ├── test_noise_estimation.py
│   ├── test_spectral_gain_estimation.py
│   ├── test_deep_learning_models.py
│   ├── test_deep_learning_common.py
│   └── test_deep_learning_registries.py
│
└── samples/                       # Sample audio files for testing
```

## Architecture

### Registry Pattern (Deep Learning)
All trainable components use a registry pattern. `train.py` reads `cfg_train.yaml`, resolves component names against registries, and instantiates them with params — no code changes needed to swap components:

1. `MODEL_REGISTRY` in `models/__init__.py` — registered: `gtcrn, crn, gcrn, gccrn, dpcrn, nsnet, df1, df2, df3`
2. `LOSS_REGISTRY` in `losses/__init__.py` — registered: `hybrid, stft, multi_stft, compressed_mse, weighted_sd, neg_snr, gain_neg_snr, sisnr`
3. `DATASET_REGISTRY` in `datasets/__init__.py` — registered: `dns3`
4. `SCHEDULER_REGISTRY` in `scheduler/__init__.py` — registered: `warmup_cosine, step, multistep, cosine, plateau`

To add a new component: (1) create the implementation file, (2) add one line to the registry in `__init__.py`, (3) reference by name in YAML config.

### Signal Processing Track
- **Noise estimators** extend `BaseNoiseEstimator` with `estimate_noise(frame_psd) -> np.ndarray`. Each tracks stateful recursion across frames (Pmin tracking, smoothing, VAD). Papers referenced in docstrings.
- **Gain estimators** extend `BaseSpectralGainEstimator` with `compute_gain(frame_psd, noise_psd) -> (gain, vad)`. Most use Decision-Directed (DD) approach for prior SNR estimation.
- Test scripts (`tests/test_noise_estimation.py`, `tests/test_spectral_gain_estimation.py`) produce output wavs in `samples/` — not pytest-based but manual verification.

### Training Pipeline
`train.py`'s `Trainer` class handles:
- Dataset build → DataLoader → epoch loop
- Per-epoch: train → validate (with PESQ via joblib parallelism)
- Checkpoint save/resume, TensorBoard logging, sample wav saving
- DDP via `torch.multiprocessing.spawn` with NCCL backend

### Key Shared Layers (`models/common/layers.py`)
- `CausalConvEncoder` / `CausalConvDecoder` — multi-stage strided convolution blocks for time-frequency processing, returns skip connections
- `complex_ratio_mask(mask, stft_spec)` — applies complex ratio masking (CRM)
- `Chomp_T` — removes trailing padding for causal convolutions

### Model Input/Output Convention
All deep learning models accept raw waveform `(batch, samples)` and return enhanced waveform of the same shape. STFT/projection is handled internally by each model.

### DeepFilterNet Models
`df1`/`df2`/`df3` require external `libdf` and `df` packages (see https://github.com/Rikorose/DeepFilterNet). They are lazily imported in the registry so non-DF models work without these dependencies.
