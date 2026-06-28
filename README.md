![noise-xorcist](./assets/icon-224.png)
# noise-xorcist
noise-xorcist is a unified single-channel speech enhancement toolbox, which incorporates a variety of traditional signal processing algorithms and modern deep learning-based methods.

## Signal processing

### noise estimation
- [Minimum statistics](./signal_processing/noise_estimation/ms.py)
- [MCRA](./signal_processing/noise_estimation/mcra.py)
- [MCRA2](./signal_processing/noise_estimation/mcra2.py)
- [IMCRA](./signal_processing/noise_estimation/imcra.py)
- [Continuous minimal tracking](./signal_processing/noise_estimation/csmt.py)
- [Weighted spectral average](./signal_processing/noise_estimation/wsa.py)
- [Connected time-frequency regions](./signal_processing/noise_estimation/cfr.py)
- [SPP](./signal_processing/noise_estimation/spp.py)

### spectral gain estimation
- [Spectral subtraction](./signal_processing/spectral_gain_estimation/spectral_subtraction.py)
- [MMSE](./signal_processing/spectral_gain_estimation/mmse.py)
- [LogMMSE](./signal_processing/spectral_gain_estimation/logmmse.py)
- [LogMMSE SPU](./signal_processing/spectral_gain_estimation/logmmse_spu.py)
- [STSA Mis](./signal_processing/spectral_gain_estimation/stsa_mis.py)
- [STSA Wcosh](./signal_processing/spectral_gain_estimation/stsa_wcosh.py)
- [STSA Weuclid](./signal_processing/spectral_gain_estimation/stsa_weuclid.py)
- [STSA Wlr](./signal_processing/spectral_gain_estimation/stsa_wlr.py)
- [Wiener](./signal_processing/spectral_gain_estimation/wiener.py)
- [OMLSA](./signal_processing/spectral_gain_estimation/omlsa.py)

## Deep learning

### network
- [NSNet](./deep_learning/models/nsnet.py)
- [CRN](./deep_learning/models/crn.py)
- [DPCRN](./deep_learning/models/dpcrn.py)
- [GCRN](./deep_learning/models/gcrn.py)
- [GCCRN](./deep_learning/models/gccrn.py)
- [GTCRN](./deep_learning/models/gtcrn.py)
- [DeepFilterNet](./deep_learning/models/deepfilternet/deepfilternet.py)
- [DeepFilterNet2](./deep_learning/models/deepfilternet/deepfilternet2.py)
- [DeepFilterNet3](./deep_learning/models/deepfilternet/deepfilternet3.py)

### loss
- [HybridLoss](./deep_learning/losses/hybrid_loss.py) (STFT magnitude + complex + SISNR)
- [WeightedSpeechDistortionLoss](./deep_learning/losses/mse_loss.py)
- [ComplexCompressedMSELoss](./deep_learning/losses/mse_loss.py)
- [STFTLoss](./deep_learning/losses/mse_loss.py)
- [MultiResolutionSTFTLoss](./deep_learning/losses/mse_loss.py)
- [NegativeSNRLoss](./deep_learning/losses/snr_loss.py)
- [GainMaskBasedNegativeSNRLoss](./deep_learning/losses/snr_loss.py)
- [SISNRLoss](./deep_learning/losses/snr_loss.py)

## Metric

### intrusive (with reference)
- [SDR](./evaluation/calculate_intrusive_se_metrics.py)
- [SISNR](./evaluation/calculate_intrusive_se_metrics.py)
- [PESQ](./evaluation/calculate_intrusive_se_metrics.py)
- [ESTOI](./evaluation/calculate_intrusive_se_metrics.py)

### non-intrusive (without reference)
- [OVRL](./evaluation/calculate_nonintrusive_dnsmos.py)
- [SIG](./evaluation/calculate_nonintrusive_dnsmos.py)
- [BAK](./evaluation/calculate_nonintrusive_dnsmos.py)
- [P808_MOS](./evaluation/calculate_nonintrusive_dnsmos.py)

## Noisyspeech synthesizer
- [SpecAugment](./noisyspeech_synthesizer/prepare_custom_datasets/augmentations.py)
- [MixAugment](./noisyspeech_synthesizer/prepare_custom_datasets/augmentations.py)
- [VolAugment](./noisyspeech_synthesizer/prepare_custom_datasets/augmentations.py)
- [ClipAugment](./noisyspeech_synthesizer/prepare_custom_datasets/augmentations.py)
- [BreakAugment](./noisyspeech_synthesizer/prepare_custom_datasets/augmentations.py)
- [HowlingAugment](./noisyspeech_synthesizer/prepare_custom_datasets/augmentations.py)
