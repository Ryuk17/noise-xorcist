# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torch
import torchaudio
from torchmetrics.audio import SignalNoiseRatio


logger = logging.getLogger(__name__)

metrics_dict = {
    "SNR": SignalNoiseRatio,
    "PESQ": PerceptualEvaluationSpeechQuality,
    "STOI": ShortTimeObjectiveIntelligibility
}

class Metrics(torch.nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.metric_list = {}

    def __build_metric(self):
        for metric_name in self.cfg["TESTS"]["METRICS"]:
            obj = metrics_dict[metric_name](self.cfg[metric_name])
            self.metric_list[metric_name] = obj

    def forward(self, denoised, clean):
        metric_scores = {}
        for metric_name in self.metric_list.keys():
            sub_loss = self.metric_list[metric_name](denoised, clean)
            metric_scores[metric_name] = sub_loss

        return metric_scores


def build_metrics(cfg):
    return Metrics(cfg)
    
