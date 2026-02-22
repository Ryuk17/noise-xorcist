'''
Author: Ryuk
Date: 2026-02-15 16:27:15
LastEditors: Ryuk
LastEditTime: 2026-02-22 18:36:06
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

    def compute_gain(self, frame_psd, noise_psd):
        """
        计算谱增益的抽象方法
        
        参数:
            frame_psd (array-like): 当前帧的功率谱
            noise_psd (array-like): 估计的噪声功率谱
            
        返回:
            gain (array-like): 计算得到的谱增益
        """
        raise NotImplementedError("子类必须实现 compute_gain 方法")
    
    