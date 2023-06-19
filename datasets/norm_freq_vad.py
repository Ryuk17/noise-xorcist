import math
import os
import glob
import numpy as np
import torch
import torchaudio
from utils.noisy_synthesizer import gen_noisy
from torch.utils.data import Dataset

torchaudio.set_audio_backend("soundfile")  # default backend (SoX) has bugs when loading WAVs

class NormFreqVAD(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.online = cfg['ONLINE_AUG']

        self.clean_dir = cfg['CLEAN_DIR']
        self.noise_dir = cfg['NOISE_DIR']
        self.noisy_dir = cfg['NOISY_DIR']
        self.rir_dir = cfg['RIR_DIR']

        self.n_fft = cfg['N_FFT']
        self.sample_rate = cfg['SAMPLE_RATE']
        self.frame_length = cfg['FRAME_LENGTH']
        self.hop_length = cfg['HOP_LENGTH']
        self.vad_freq_floor = cfg['VAD_FREQ_FLOOR']
        self.vad_freq_ceil = cfg['VAD_FREQ_CEIL']

        if cfg['FFT_WINDOW'] == 'HAMMING':
            self.fft_win = torch.hamming_window(self.n_fft)
        elif cfg['FFT_WINDOW'] == 'HANNING':
            self.fft_win = torch.hanning_window(self.n_fft)
        else:
            self.fft_win = torch.ones(self.n_fft)

        self.clean_wav = sorted(glob.glob(f"{self.clean_dir}/*.wav"))
        if not self.online:
            assert self.noisy_dir is not None, "noisy dir cannot be None when for online processing"
            self.noisy_wav =  sorted(glob.glob(f"{self.noisy_dir}/*.wav"))
        else:
            self.noise_wav = glob.glob(f"{self.noise_dir}/*.wav")
            if self.rir_dir is not None:
                self.rir_wav = glob.glob(f"{self.rir_dir}/*.wav")

        # vad
        step = self.sample_rate / self.n_fft
        frequency_bins = np.arange(0, (self.n_fft // 2 + 1) * step, step=step)
        self.vad_frequencies = np.where((frequency_bins >= self.vad_freq_floor) & (frequency_bins <= self.vad_freq_ceil), True, False)

        # mean normalization
        frameshift = (self.n_fft / self.sample_rate) / 4  # frameshift between STFT frames
        t_init = 0.1  # init time
        tauFeat = 3  # time constant
        tauFeatInit = 0.1  # time constant during init
        self.n_init_frames = math.ceil(t_init / frameshift)
        self.alpha_feat_init = math.exp(-frameshift / tauFeatInit)
        self.alpha_feat = math.exp(-frameshift / tauFeat)

    def __len__(self):
        return len(self.clean_wav)

    def __getitem__(self, idx):
        clean_path = self.clean_wav[idx]
        clean_waveform, _ = torchaudio.load(clean_path, normalization=2 ** 15)

        if self.online:
            noise_path = self.noise_wav[idx]
            noise_waveform, _ = torchaudio.load(noise_path, normalization=2 ** 15)

            rir_waveform = None
            if self.rir_dir is not None:
                rir_path = self.rir_wav[idx]
                rir_waveform, _ = torchaudio.load(rir_path, normalization=2 ** 15)
            noisy_waveform = gen_noisy_online(self.cfg, clean_waveform, noise_waveform, rir_waveform)
        else:
            noisy_path = self.noisy_wav[idx]
            noisy_waveform, _ = torchaudio.load(noisy_path, normalization=2 ** 15)

        assert clean_waveform.shape[0] == 1 and noisy_waveform.shape[0] == 1, 'WAV file is not single channel!'

        x_stft = torch.stft(noisy_waveform.view(-1), n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.n_fft, window=self.fft_win)
        y_stft = torch.stft(clean_waveform.view(-1), n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.n_fft, window=self.fft_win)

        x_ps = x_stft.pow(2).sum(-1)
        x_lps = LogTransform()(x_ps)

        x_ms = x_ps.sqrt()
        y_ms = y_stft.pow(2).sum(-1).sqrt()

        noise_ms = (x_stft - y_stft).pow(2).sum(-1).sqrt()

        # VAD
        y_ms_filtered = y_ms[self.vad_frequencies]
        y_energy_filtered = y_ms_filtered.pow(2).mean(dim=0)
        y_energy_filtered_averaged = self.__moving_average(y_energy_filtered)
        y_peak_energy = y_energy_filtered_averaged.max()
        vad = torch.where(y_energy_filtered_averaged > y_peak_energy / 1000, torch.ones_like(y_energy_filtered),
                          torch.zeros_like(y_energy_filtered))
        vad = vad.bool()

        # mean normalization
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

        x_lps = torch.stack(frames, dim=0).transpose(0, 1)  # (frequency, time)

        if not self.test:
            return x_lps, x_ms, y_ms, noise_ms, vad
        if self.test:
            return noisy_waveform.view(-1), clean_waveform.view(-1), x_stft, y_stft, x_lps, x_ms, y_ms, VAD

    def __moving_average(self, a, n=3):
        ret = torch.cumsum(a, dim=0)
        ret[n:] = ret[n:] - ret[:-n]
        ret[:n - 1] = a[:n - 1]
        ret[n - 1:] = ret[n - 1:] / n
        return ret


class LogTransform(torch.nn.Module):
    def __init__(self, floor=10 ** -12):
        super().__init__()
        self.floor = floor

    def forward(self, specgram):
        return torch.log(torch.clamp(specgram, min=self.floor))