'''
Author: Ryuk
Date: 2026-02-15 16:27:15
LastEditors: Ryuk
LastEditTime: 2026-02-15 17:02:52
Description: First create
'''



class BaseNoiseEstimator:
    """
    噪声估计基类
    """

    def __init__(self):
        pass

    def estimate_noise(self, signal):
        """
        估计噪声的抽象方法
        
        参数:
            signal (array-like): 输入信号
            
        返回:
            noise_estimate (array-like): 估计的噪声
        """
        raise NotImplementedError("子类必须实现 estimate_noise 方法")
    


class BaseSpectralGainEstimator:
    """
    谱增益计算基类
    """

    def __init__(self):
        pass

    def compute_gain(self, signal, noise_estimate):
        """
        计算谱增益的抽象方法
        
        参数:
            signal (array-like): 输入信号
            noise_estimate (array-like): 估计的噪声
            
        返回:
            gain (array-like): 计算得到的谱增益
        """
        raise NotImplementedError("子类必须实现 compute_gain 方法")
    
    

class BaseDenoiser:
    """
    组合降噪算法基类
    """

    def __init__(self, noise_estimator: BaseNoiseEstimator, spectral_gain: BaseSpectralGainEstimator):
        """
        初始化降噪算法
        
        参数:
            noise_estimator (BaseNoiseEstimator): 噪声估计器实例
            spectral_gain (BaseSpectralGain): 谱增益计算器实例
        """
        self.noise_estimator = noise_estimator
        self.spectral_gain = spectral_gain

    def denoise(self, signal):
        """
        执行降噪的主方法
        
        参数:
            signal (array-like): 输入信号
            
        返回:
            denoised_signal (array-like): 降噪后的信号
        """
        # 步骤1：估计噪声
        noise_estimate = self.noise_estimator.estimate_noise(signal)

        # 步骤2：计算谱增益
        gain = self.spectral_gain.compute_gain(signal, noise_estimate)

        # 步骤3：应用谱增益进行降噪
        denoised_signal = self._apply_gain(signal, gain)

        return denoised_signal

    def _apply_gain(self, signal, gain):
        """
        应用谱增益到信号上（可被子类重写）
        
        参数:
            signal (array-like): 输入信号
            gain (array-like): 谱增益
            
        返回:
            result (array-like): 应用增益后的结果
        """
        # 默认实现为逐元素相乘
        return signal * gain