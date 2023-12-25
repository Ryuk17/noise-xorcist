# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
import time
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from noisexorcist.utils import comm
from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class SeEvaluator(DatasetEvaluator):
    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._cpu_device = torch.device('cpu')
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        prediction = {
            'masks': outputs.to(self._cpu_device, torch.float32),
            'feats': inputs['feats'].to(self._cpu_device),
        }
        self._predictions.append(prediction)

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        for prediction in predictions:
            pass

        self._results = OrderedDict()


        return copy.deepcopy(self._results)

