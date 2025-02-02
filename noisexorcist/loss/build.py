# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging
from mse_loss import *
from phase_loss import *
from snr_loss import *


logger = logging.getLogger(__name__)

losses_dict = {
    "WSD": WeightedSpeechDistortionLoss
}


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.loss_list = {}

    def __build_loss(self):
        for loss_name in self.cfg["NAME"]:
            obj = losses_dict[loss_name](self.cfg[loss_name])
            self.loss_list[loss_name] = [obj, self.cfg[loss_name]["SCALE"]]

    def forward(self, inputs, data):
        loss_dict = {}
        for loss_name in self.loss_list.keys():
            sub_loss = self.loss_list[loss_name][0](inputs, data) * self.loss_list[loss_name][1]
            loss_dict[loss_name] = sub_loss

        return loss_dict



def build_loss(cfg):
    loss_obj = Loss(cfg)
    return loss_obj