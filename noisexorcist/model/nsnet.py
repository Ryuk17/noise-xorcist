# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging
import math

import torch
import torch.nn as nn

from noisexorcist.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

logger = logging.getLogger(__name__)


class NSNetModel(nn.Module):
    def __init__(self, input_dim, n_gru_layers, gru_dropout):
        super(NSNetModel, self).__init__()
        self.n_frequency_bins = input_dim
        self.n_gru_layers = n_gru_layers
        self.gru_dropout = gru_dropout

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """
        self.gru = nn.GRU(input_size=self.n_frequency_bins, hidden_size=self.n_frequency_bins, num_layers=self.n_gru_layers,
                          batch_first=True, dropout=self.gru_dropout)
        self.dense = nn.Linear(in_features=self.n_frequency_bins, out_features=self.n_frequency_bins)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, time, n_frequency_bins)
        x, _ = self.gru(x)  # (batch_size, time, n_frequency_bins)
        x = torch.sigmoid(self.dense(x))  # (batch_size, time, frequency_bins)
        x = x.permute(0, 2, 1)  # (batch_size, frequency_bins, time)

        return x


def build_nsnet(cfg):
    # fmt: off
    input_dim = cfg["INPUT_DIM"]
    n_gru_layers = cfg["GRU_LAYERS"]
    gru_dropout = cfg["GRU_DROPOUT"]
    pretrain_path = cfg["PRETRAIN_PATH"]
    # fmt: on

    model = NSNetModel(input_dim, n_gru_layers, gru_dropout)

    if pretrain_path:
        try:
            state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
            logger.info(f"Loading pretrained model from {pretrain_path}")
        except FileNotFoundError as e:
            logger.info(f'{pretrain_path} is not found! Please check this path.')
            raise e
        except KeyError as e:
            logger.info("State dict keys error! Please check the state dict.")
            raise e

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    model.to(torch.device(cfg["DEVICE"]))
    return model