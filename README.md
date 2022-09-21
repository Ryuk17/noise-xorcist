# SEAugment
Data augmentations for speech enhancement


## What's in it
+ **[SpecTransform](https://github.com/Ryuk17/SEAugment/blob/main/src/spec_aug.py)**
was proposed in [RNNoise](https://github.com/xiph/rnnoisehttps://github.com/xiph/rnnoise), which is achieved by filtering the noise and speech signal independently for each training example using a second order filter  
![SpecTransform](https://github.com/Ryuk17/SEAugment/blob/main/assets/spec_trans.png)

+ **[MixTransform](https://github.com/Ryuk17/SEAugment/blob/main/src/mix_aug.py)**
uses different snr combine speech samples and noise samples, which is a common method for data augment in speech enhencement  
![MixTransform](https://github.com/Ryuk17/SEAugment/blob/main/assets/mix_trans.png)

+ **[VolTransform](https://github.com/Ryuk17/SEAugment/blob/main/src/vol_aug.py)**
apply step gains to target audio, which simulates different microphone volumes  
![VolTransform](https://github.com/Ryuk17/SEAugment/blob/main/assets/vol_trans.png)

+ **[FilterTransform]()**
design a biquad peaking equalizer filter and perform filtering on samples, which simulates different microphone frequency response  
![FilterTransform](https://github.com/Ryuk17/SEAugment/blob/main/assets/filter_trans.png)

+ **[ClipTransform]()**

+ **[ReverbTransform]()**

+ **[BreakTransform]()**

+ **[HowlingTransform]()**

+ **[DynamicTransform]()**

