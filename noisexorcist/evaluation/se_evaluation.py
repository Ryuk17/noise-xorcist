# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import copy
import itertools
import logging
from collections import OrderedDict

import torch

from noisexorcist.utils import comm
from noisexorcist.data import restore_waveform
from .evaluator import DatasetEvaluator
from .metrics import build_metrics

logger = logging.getLogger(__name__)


class SeEvaluator(DatasetEvaluator):
    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg
        self.metrics = build_metrics(self.cfg)
        self._output_dir = output_dir
        self._cpu_device = torch.device('cpu')
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, data, model_outputs):
        denoised, clean = restore_waveform(self.cfg, data, model_outputs)
        denoised = denoised.to(self._cpu_device, torch.float32)
        clean = clean.to(self._cpu_device, torch.float32)

        results = self.metrics(denoised, clean)
        results["num_samples"] = model_outputs.size(0)
        self._predictions.append(results)

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process(): return {}

        else:
            predictions = self._predictions

        total_samples = 0
        total_snr = 0
        total_pesq = 0
        total_stoi = 0
        for prediction in predictions:
            total_samples += prediction["num_samples"]
            total_snr += prediction["snr"]
            total_pesq += prediction["pesq"]
            total_stoi += prediction["stoi"]

        snr = total_snr / total_samples
        pesq = total_pesq / total_samples
        stoi = total_stoi / total_samples

        self._results = OrderedDict()
        self._results["snr"] = snr
        self._results["pesq"] = pesq
        self._results["stoi"] = stoi

        return copy.deepcopy(self._results)