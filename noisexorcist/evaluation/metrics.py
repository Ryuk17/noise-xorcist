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
        self.nb_pesq = PerceptualEvaluationSpeechQuality(self.cfg["DATA"]["SAMPLE_RATE"], 'wb')
        self.stoi = ShortTimeObjectiveIntelligibility(self.cfg["DATA"]["SAMPLE_RATE"], False)
        self.snr = SignalNoiseRatio()

    def forward(self, denoised, clean):
        pesq_scores, stoi_scores, snr_scores = 0, 0, 0
        for denoised_wav, clean_wav in zip(denoised, clean):
            try:
                pesq_scores += self.nb_pesq(denoised_wav, clean_wav).item()
                stoi_scores += self.stoi(denoised_wav, clean_wav).item()
                snr_scores += self.snr(denoised_wav, clean_wav).item()
            except ValueError as e:
                logger.info("Error {} when calculate metrics".format(e))
                raise

        return {
            'snr': snr_scores,
            'pesq': pesq_scores,
            'stoi': stoi_scores
        }


def build_metrics(cfg):
    return Metrics(cfg)
    
