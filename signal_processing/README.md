<!--
 * @Author: Ryuk
 * @Date: 2026-02-15 16:27:15
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-06-28
 * @Description: Traditional signal processing for single-channel speech enhancement.
-->

## Directory Structure

| Directory | Description |
|-----------|-------------|
| `base/` | Abstract base classes (`BaseNoiseEstimator`, `BaseSpectralGainEstimator`) |
| `noise_estimation/` | Noise PSD estimators, one class per file |
| `spectral_gain_estimation/` | Spectral gain / suppression rule estimators, one class per file |

## Design

The module follows a **composable two-stage pipeline**:

1. **Noise estimation** — given the noisy power spectrum, estimate the noise PSD
2. **Gain estimation** — given the noisy and noise PSDs, compute a suppression gain

These two stages are fully decoupled through abstract base classes in [base/denoiser.py](base/denoiser.py):

- `BaseNoiseEstimator.estimate_noise(frame_psd)` → `noise_psd: np.ndarray`
- `BaseSpectralGainEstimator.compute_gain(frame_psd, noise_psd)` → `(gain, vad)`

Any noise estimator can be paired with any gain estimator. The typical frame-level processing loop is:

```python
for frame in frames:
    spectrum  = np.fft.rfft(frame * window)
    frame_psd = np.abs(spectrum) ** 2
    noise_psd = noise_estimator.estimate_noise(frame_psd)
    gain, _   = gain_estimator.compute_gain(frame_psd, noise_psd)
    enhanced  = np.fft.irfft(spectrum * gain)
    output    = overlap_add(enhanced * window)
```

## Noise Estimation

All estimators extend `BaseNoiseEstimator` and maintain internal state for recursive tracking across frames.

| File | Class | Algorithm | Reference |
|------|-------|-----------|-----------|
| `ms.py` | `MSNoiseEstimator` | Minimum Statistics | Martin (2001) |
| `mcra.py` | `MCRANoiseEstimator` | Minima Controlled Recursive Averaging | Rangachari & Loizou (2006) |
| `mcra2.py` | `MCRA2NoiseEstimator` | MCRA v2 | Cohen (2002) |
| `imcra.py` | `IMCRANoiseEstimator` | Improved MCRA | Cohen (2003) |
| `csmt.py` | `CSMTNoiseEstimator` | Continuous Spectral Minimum Tracking | Doblinger (1995) |
| `wsa.py` | `WSANoiseEstimator` | Weighted Spectral Average | Hirsch & Ehrlicher (1995) |
| `cfr.py` | `CFRNoiseEstimator` | Connected Time-Frequency Regions | Sørensen & Andersen (2005) |
| `spp.py` | `SPPNoiseEstimator` | Speech Presence Probability | Gerkmann & Hendriks (2012) |

All constructors take `n_fft` as the first parameter and expose `self.n_fft` for consistency checks with the gain estimator.

## Spectral Gain Estimation

All estimators extend `BaseSpectralGainEstimator` and implement `compute_gain(frame_psd, noise_psd)`. Most use the **Decision-Directed** approach for prior SNR estimation.

| File | Class | Algorithm | Reference |
|------|-------|-----------|-----------|
| `spectral_subtraction.py` | `SSSpectralGainEstimator` | Spectral Subtraction (Berouti / simple) | — |
| `mmse.py` | `MMSESpectralEstimator` | MMSE-STSA | Ephraim & Malah (1985) |
| `logmmse.py` | `LogMMSESpectralEstimator` | Log-MMSE | Ephraim & Malah (1985) |
| `logmmse_spu.py` | `LogMMSESpuSpectralEstimator` | Log-MMSE with SPU | Ephraim & Malah (1985) |
| `stsa_mis.py` | `STSAMisSpectralGainEstimator` | STSA — MIS distance | — |
| `stsa_wcosh.py` | `STSAWCoshSpectralGainEstimator` | STSA — weighted cosh | — |
| `stsa_weuclid.py` | `STSAWeuclidSpectralGainEstimator` | STSA — weighted Euclidean | — |
| `stsa_wlr.py` | `STSAWlrSpectralGainEstimator` | STSA — weighted likelihood ratio | — |
| `wiener.py` | `WienerSpectralGainEstimator` | Wiener filter | Scalart & Filho (1996) |
| `omlsa.py` | `OMLSASpectralGainEstimator` | OM-LSA | Cohen & Berdugo (2001) |

## Usage

### Quick Start

```python
from signal_processing.noise_estimation import MCRANoiseEstimator
from signal_processing.spectral_gain_estimation import SSSpectralGainEstimator

noise_est = MCRANoiseEstimator(n_fft=512)
gain_est = SSSpectralGainEstimator(n_fft=512, mode="berouti")

for frame in frames:
    frame_psd = np.abs(np.fft.rfft(frame * window)) ** 2
    noise_psd = noise_est.estimate_noise(frame_psd)
    gain, _ = gain_est.compute_gain(frame_psd, noise_psd)
    # apply gain...
```

### Run Tests

```bash
python tests/test_noise_estimation.py          # all 8 noise estimators
python tests/test_spectral_gain_estimation.py  # gain estimators with MCRA
```

These write output `.wav` files to `samples/` for manual listening.

### Run Example

```bash
python examples/mcra_spectral_substraction.py
```

## Adding a New Component

Two steps, following the existing pattern:

1. Create the implementation file in the appropriate directory:

```python
# noise_estimation/my_estimator.py
from ..base import BaseNoiseEstimator

class MyNoiseEstimator(BaseNoiseEstimator):
    def __init__(self, n_fft, ...):
        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1
        # state variables...

    def estimate_noise(self, frame_psd):
        # update state, return noise_psd
        return noise_psd
```

2. Export it in the package `__init__.py`:

```python
# noise_estimation/__init__.py
from .my_estimator import MyNoiseEstimator
__all__.append("MyNoiseEstimator")
```

The same pattern applies for gain estimators — extend `BaseSpectralGainEstimator`, implement `compute_gain(frame_psd, noise_psd) → (gain, vad)`, and export.

## Notes

1. All processing is in the frequency domain. Input signals are expected to be pre-framed and windowed.
2. The pipeline does not include VAD, framing, or overlap-add — these are handled by the caller (see `examples/`).
3. Numerical stability: most implementations use `eps=1e-12` to guard against division by zero and clamp gain to `[0, 1]`.
