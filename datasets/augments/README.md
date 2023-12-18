# torch-se
Data augmentations for speech enhancement


## What's in it
+ **[SpecTransform](https://github.com/Ryuk17/torch-se/blob/main/src/spec_aug.py)**
was proposed in [RNNoise](https://github.com/xiph/rnnoisehttps://github.com/xiph/rnnoise), which is achieved by filtering the noise and speech signal independently for each training example using a second order filter  
![SpecTransform](https://github.com/Ryuk17/torch-se/blob/main/assets/spec_trans.png)

+ **[MixTransform](https://github.com/Ryuk17/torch-se/blob/main/src/mix_aug.py)**
uses different snr combine speech samples and noise samples, which is a common method for data augment in speech enhencement  
![MixTransform](https://github.com/Ryuk17/torch-se/blob/main/assets/mix_trans.png)

+ **[VolTransform](https://github.com/Ryuk17/torch-se/blob/main/src/vol_aug.py)**
apply step gains to target audio, which simulates different microphone volumes  
![VolTransform](https://github.com/Ryuk17/torch-se/blob/main/assets/vol_trans.png)

+ **[FilterTransform]()**
design a biquad peaking equalizer filter and perform filtering on samples, which simulates different microphone frequency response  
![FilterTransform](https://github.com/Ryuk17/torch-se/blob/main/assets/filter_trans.png)

+ **[ClipTransform]()**
truncate samples whose amplitude larger than a given level, which simulates clipping effect  
![ClipTransform](https://github.com/Ryuk17/torch-se/blob/main/assets/clip_trans.png)

+ **[ReverbTransform]()**
uses convolve to simulates reverberation. RIR datasets: [OpenSLR26](http://www.openslr.org/26/) and [OpenSLR28](http://www.openslr.org/28/)  
![ReverbTransform](https://github.com/Ryuk17/torch-se/blob/main/assets/reverb_trans.png)


+ **[BreakTransform]()**
use time-axis mask to simulates frame drop in communication  
![BreakTransform](https://github.com/Ryuk17/torch-se/blob/main/assets/break_trans.png)


+ **[HowlingTransform]()**
use IR and feedback to generate howling effect  
![HowlingTransform](https://github.com/Ryuk17/torch-se/blob/main/assets/howling_trans.png)

+ **[DynamicTransform]()**

