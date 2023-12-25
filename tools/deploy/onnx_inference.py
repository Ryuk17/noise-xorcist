# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os

import numpy as np
import onnxruntime
import tqdm
import librosa
import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser(description="onnx model inference")

    parser.add_argument(
        "--model-path",
        default="onnx_model/baseline.onnx",
        help="onnx model path"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='onnx_output',
        help='path to save converted caffe model'
    )
    return parser




if __name__ == "__main__":
    args = get_parser().parse_args()

    ort_sess = onnxruntime.InferenceSession(args.model_path)

    input_name = ort_sess.get_inputs()[0].name

    if not os.path.exists(args.output): os.makedirs(args.output)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):
            wav, sr = librosa.load(path, sr=None)
            feat = ort_sess.run(None, {input_name: wav})[0]
            sf.write()
