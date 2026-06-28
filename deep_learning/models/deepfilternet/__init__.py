"""
DeepFilterNet v1/v2/v3 模型。
通过各自模块内置的 init_model() 封装 ModelParams / ERB / DF 初始化。
"""
from .deepfilternet import init_model as init_df1
from .deepfilternet2 import init_model as init_df2
from .deepfilternet3 import init_model as init_df3
