# SEAugment
Data augmentations for speech enhancement


## What's in it
+ **[SpecTransform](https://github.com/Ryuk17/SEAugment/blob/main/src/spec_aug.py)**
SpecTransform was proposed in [RNNoise](https://github.com/xiph/rnnoisehttps://github.com/xiph/rnnoise), which is achieved by filtering the noise and speech signal independently for each training example using a second order filter
![SpecTransform](https://github.com/Ryuk17/SEAugment/blob/main/assets/spec_trans.png)


+ **[MixTransform](https://github.com/Ryuk17/SEAugment/blob/main/src/mix_aug.py)**
MixTransform use different snr combine speech samples and noise samples, which is a common method for data augment in speech enhencement  
![MixTransform](https://github.com/Ryuk17/SEAugment/blob/main/assets/mix_trans.png)

+ **[VolTransform](https://github.com/Ryuk17/SEAugment/blob/main/src/vol_aug.py)**
VolTransform use step gains to process target audio, which simulates different microphone volumes  
![VolTransform](https://github.com/Ryuk17/SEAugment/blob/main/assets/vol_trans.png)

+ **[FilterTransform]()**

+ **[ClipTransform]()**

+ **[ReverbTransform]()**

+ **[BreakTransform]()**

+ **[HowlingTransform]()**

+ **[DynamicTransform]()**

