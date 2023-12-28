"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import numpy as np
import torch
import math
from torch.utils.data import Dataset
import torchaudio
import os
from .vad import VadDetector


class Spectrum(Dataset):
    """
    Create a PyTorch Dataset object from a directory containing clean and noisy WAV files
    """
    def __init__(self, cfg, mode, verbose=True, **kwargs):
        self.clean_dir = cfg['DATA']['CLEAN_DIR']
        self.noisy_dir = cfg['DATA']['NOISY_DIR']
        self.sample_rate = cfg['INPUT']['SAMPLE_RATE']
        self.frame_len = cfg['INPUT']['FRAME_LEN']
        self.nfft = cfg['INPUT']['NFFT']
        self.hop_len = cfg['INPUT']['HOP_LEN']
        self.win_type = cfg['INPUT']['WIN_TYPE']
        self.verbose = verbose
        self.mode = mode
        self.normal = cfg['INPUT']['NORMAL']
        self.vad_detector = VadDetector(self.sample_rate, self.nfft)

        if self.win_type == "hamming":
            self.window = torch.hamming_window(self.nfft)
        elif self.win_type == "hanning":
            self.window = torch.hann_window(self.nfft)
        else:
            raise NotImplementedError

        assert os.path.exists(self.clean_dir), 'No clean WAV file folder found!'
        assert os.path.exists(self.noisy_dir), 'No noisy WAV file folder found!'

        self.clean_wav_list = {}
        for i, filename in enumerate(sorted(os.listdir(self.clean_dir))):
            self.clean_wav_list[i] = self.clean_dir.joinpath(filename)

        self.noisy_wav_list = {}
        for i, filename in enumerate(sorted(os.listdir(self.noisy_dir))):
            self.noisy_wav_list[i] = self.noisy_dir.joinpath(filename)

        if self.normal:
            frameshift = (self.nfft / self.sample_rate) / 4  # frameshift between STFT frames
            t_init = 0.1  # init time
            tauFeat = 3  # time constant
            tauFeatInit = 0.1  # time constant during init
            self.n_init_frames = math.ceil(t_init / frameshift)
            self.alpha_feat_init = math.exp(-frameshift / tauFeatInit)
            self.alpha_feat = math.exp(-frameshift / tauFeat)

    def __len__(self):
        return len(self.noisy_wav_list)

    def __getitem__(self, idx):
        noisy_path = self.noisy_WAVs[idx]
        clean_path = self.clean_dir.joinpath(noisy_path.name.split('.')[0] + '.wav')  # get the filename of the clean WAV from the filename of the noisy WAV
        while True:
            try:
                clean_waveform, _ = torchaudio.load(clean_path, normalization=2**15)
                noisy_waveform, _ = torchaudio.load(noisy_path, normalization=2**15)
            except (RuntimeError, OSError):
                continue
            break

        assert clean_waveform.shape[0] == 1 and noisy_waveform.shape[0] == 1, 'WAV file is not single channel!'

        x_stft = torch.stft(noisy_waveform.view(-1), nfft=self.nfft, hop_length=self.hop_len, win_length=self.nfft, window=self.window)
        y_stft = torch.stft(clean_waveform.view(-1), nfft=self.nfft, hop_length=self.hop_len, win_length=self.nfft, window=self.window)

        x_ps = x_stft.pow(2).sum(-1)
        x_lps = LogTransform()(x_ps)

        y_ps = y_stft.pow(2).sum(-1)
        y_lps = LogTransform()(y_ps)

        x_ms = x_ps.sqrt()
        y_ms = y_ps.sqrt()

        noise_ms = (x_stft - y_stft).pow(2).sum(-1).sqrt()

        vad = self.vad_detector(y_ms)

        if self.normal:
            frames = []
            x_lps = x_lps.transpose(0, 1)  # (time, frequency)
            n_init_frames = self.n_init_frames
            alpha_feat_init = self.alpha_feat_init
            alpha_feat = self.alpha_feat
            for frame_counter, frame_feature in enumerate(x_lps):
                if frame_counter < n_init_frames:
                    alpha = alpha_feat_init
                else:
                    alpha = alpha_feat
                if frame_counter == 0:
                    mu = frame_feature
                    sigmasquare = frame_feature.pow(2)
                mu = alpha * mu + (1 - alpha) * frame_feature
                sigmasquare = alpha * sigmasquare + (1 - alpha) * frame_feature.pow(2)
                sigma = torch.sqrt(torch.clamp(sigmasquare - mu.pow(2), min=1e-12))  # limit for sqrt
                norm_feature = (frame_feature - mu) / sigma
                frames.append(norm_feature)
            x_lps = torch.stack(frames, dim=0).transpose(0, 1)   # (frequency, time)

        if self.mode == "train":
            train_dict = {
                "x_stft": x_stft,
                "y_stft": y_stft,
                "x_lps": x_lps,
                "y_lps": y_lps,
                "x_ms": x_ms,
                "y_ms": y_ms,
                "noise_ms": noise_ms,
                "vad": vad
            }
            return train_dict
        if self.mode == "test":
            test_dict = {
                "noisy_waveform": noisy_waveform.view(-1),
                "clean_waveform": clean_waveform.view(-1),
                "x_stft": x_stft,
                "y_stft": y_stft,
                "x_lps": x_lps,
                "y_lps": y_lps,
                "x_ms": x_ms,
                "y_ms": y_ms,
                "vad": vad
            }
            return test_dict

    def __moving_average(self, a, n=3):
        ret = torch.cumsum(a, dim=0)
        ret[n:] = ret[n:] - ret[:-n]
        ret[:n - 1] = a[:n - 1]
        ret[n - 1:] = ret[n - 1:] / n
        return ret


class LogTransform(torch.nn.Module):
    def __init__(self, floor=10**-12):
        super().__init__()
        self.floor = floor

    def forward(self, specgram):
        return torch.log(torch.clamp(specgram, min=self.floor))