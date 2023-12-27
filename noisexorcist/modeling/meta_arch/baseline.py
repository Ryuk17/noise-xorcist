# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import torch
from torch import nn

from noisexorcist.config import configurable
from noisexorcist.modeling.backbones import build_backbone
from noisexorcist.modeling.losses import *


class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
        """
        super().__init__()
        # backbone
        self.backbone = backbone
        self.loss_kwargs = loss_kwargs

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            'backbone': backbone,
            'input_feats': cfg.INPUT.FEATURES,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'mse': {
                        'eps': cfg.MODEL.LOSSES.MSE.EPSILON,
                    },
                    'wmse': {
                        'alpha': cfg.MODEL.LOSSES.WMSE.ALPHA,
                    },
                    'snr': {
                        'eps': cfg.MODEL.LOSSES.SNR.EPS,
                    },
                    'ci_sdr': {
                        'filter_length': cfg.MODEL.LOSSES.CI_SDR.FILTER_LENGTH,
                    }
                }
        }

    def forward(self, batched_inputs):
        inputs = self.preprocess_inputs(batched_inputs)

        outputs = self.backbone(inputs)
        if self.training:
            losses = self.losses(outputs, inputs)
            return losses
        else:
            return outputs


    def preprocess_inputs(self, batched_inputs):
        """
        get the input features.
        """
        x_stft, y_stft, x_lps, y_lps, x_ms, y_ms, noise_ms, vad = batched_inputs

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, batched_inputs):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict
