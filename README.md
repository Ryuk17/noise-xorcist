![noise-xorcist](./assets/icon-224.png)
# noise-xorcist 
noise-xorcist is a unified single-channel speech enhancement toolbox, which incorporates a variety of traditional signal processing algorithms and modern deep learning-based methods. Notice: The code is still under development and some parts not yet ready for use.

## Singal processing
  
### noise estimitation  
- [Minimum statistics](./signal_processing/noise_estimation/ms.py) 
- [MCRA](./signal_processing/noise_estimation/mcra.py) 
- [MCRA2](./signal_processing/noise_estimation/mcra2.py) 
- [IMCRA](./signal_processing/noise_estimation/imcra.py) 
- [Continuous minimal tracking](./signal_processing/noise_estimation/csmt.py) 
- [Weighted spectral average](./signal_processing/noise_estimation/wsa.py) 
- [Connected time-frequency regions](./signal_processing/noise_estimation/cfr.py) 
- [SPP](./signal_processing/noise_estimation/spp.py) 


### spectral gain estimation
- [Spectral substractive](./signal_processing/spectral_gain_estimation/spectral_subtractive.py)
- [MMSE](./signal_processing/spectral_gain_estimation/mmse.py)
- [LogMMSE](./signal_processing/spectral_gain_estimation/logmmse.py)
- [LogMMSE SPU](./signal_processing/spectral_gain_estimation/logmmse_spu.py)
- [STSA Mis](./signal_processing/spectral_gain_estimation/stat_mis.py)
- [STSA Wcosh](./signal_processing/spectral_gain_estimation/stat_wcosh.py)
- [STSA Weuclid](./signal_processing/spectral_gain_estimation/stat_weuclid.py)
- [STSA Wlr](./signal_processing/spectral_gain_estimation/stat_wlr.py)
- [Wiener](./signal_processing/spectral_gain_estimation/wiener.py)
- [Omlsa](./signal_processing/spectral_gain_estimation/omlsa.py)


## Deep learning

### network
- [NSNet](./deep_learning/models/nsnet.py)
- [CRN](./deep_learning/models/crn.py)
- [DPCRN](./deep_learning/models/dpcrn.py)
- [GCRN](./deep_learning/models/gcrn.py)
- [GCCRN](./deep_learning/models/gccrn.py)
- [GTCRN](./deep_learning/models/gtcrn.py)
- [DeepfilterNet](./deep_learning/models/deepfilternet.py)
- [DeepfilterNet2](./deep_learning/models/deepfilternet2.py)
- [DeepfilterNet3](./deep_learning/models/deepfilternet3.py)

### loss
- [WeightedSpeechDistortionLoss](./deep_learning/losses/mse_loss.py)
- [ComplexCompressedMSELoss](./deep_learning/losses/mse_loss.py)
- [NegativeSNRLoss](./deep_learning/losses/snr_loss.py)
- [GainMaskBasedNegativeSNRLoss](./deep_learning/losses/snr_loss.py)
- [STFTLoss](./deep_learning/losses/mse_loss.py)
- [MultiResolutionSTFTLoss](./deep_learning/losses/mse_loss.py)
- [SISNRLoss](./deep_learning/losses/snr_loss.py)

## noisyspeech synthesizer
- [SpecAugment](./noisyspeech_synthesizer/augmentations.py)
- [MixAugment](./noisyspeech_synthesizer/augmentations.py)
- [VolAugment](./noisyspeech_synthesizer/augmentations.py)
- [ClipAugment](./noisyspeech_synthesizer/augmentations.py)
- [BreakAugment](./noisyspeech_synthesizer/augmentations.py)
- [HowlingAugment](./noisyspeech_synthesizer/augmentations.py)
