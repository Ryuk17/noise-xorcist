# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .nsnet import build_nsnet_backbone

__all__ = {
    "build_nsnet_backbone": build_nsnet_backbone
}

def build_backbone(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone`
    """

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = __all__[backbone_name](cfg)
    return backbone
