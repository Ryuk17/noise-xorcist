"""
@FileName: __init__.py
@Description: Implement __init__
@Author: Ryuk
@CreateDate: 2022/09/20
@LastEditTime: 2022/09/20
@LastEditors: Please set LastEditors
@Version: v0.1
"""

from .gru import GruNet

model_dict = {
    'gru': GruNet
}

def get_model(cfg):
    model_name = cfg['MODEL']['MODEL_NAME']
    assert model_name in model_dict.keys(), f"There is no model named {model_name}"
    return model_dict['model_name'](cfg)
