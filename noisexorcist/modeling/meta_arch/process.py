# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""


def extract_inputs(batched_inputs, inputs_type):
    if inputs_type == "LogPowerSpectrum":
        return batched_inputs["x_lps"]
    elif inputs_type == "MagnitudeSpectrum":
        return batched_inputs["x_ms"]
    else:
        raise NotImplementedError


def extract_groundtruth(batched_inputs, gt_type):
    pass