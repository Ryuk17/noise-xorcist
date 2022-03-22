# SEAugment
Data augment for speech enhancement


## What's in it
+ **[SpecTransform](https://github.com/Ryuk17/SEAugment/blob/d2c6f4aa0c4a62a5e0f0641d76cfc907de86da37/se_augment.py#L16)**  
SpecTransform was proposed in [RNNoise](https://github.com/xiph/rnnoisehttps://github.com/xiph/rnnoise), which is achieved by filtering the noise and
speech signal independently for each training example using a second order filter of the form  
![image](https://user-images.githubusercontent.com/22525811/159518560-8ed13625-21c4-40c5-b07a-4655c3b80f36.png)


