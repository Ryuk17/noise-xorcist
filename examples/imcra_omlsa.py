'''
Author: Ryuk
Date: 2026-02-15 17:10:24
LastEditors: Ryuk
LastEditTime: 2026-02-22 18:45:25
Description: IMCRA 噪声估计 + OMLSA 谱增益, 批量推理 scp 文件
'''

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from signal_processing.noise_estimation import IMCRANoiseEstimator
from signal_processing.spectral_gain_estimation import OMLSASpectralGainEstimator


class IMCRA_OMLSA:
    def __init__(self, noise_estimator, spectral_gain_estimator, n_fft=256, frame_len=256, hop_len=128, win_type="hamming", eps=1e-12):
        self.noise_estimator = noise_estimator
        self.spectral_gain_estimator = spectral_gain_estimator

        self.frame_len = frame_len
        self.hop_len = hop_len

        assert noise_estimator.n_fft == spectral_gain_estimator.n_fft == n_fft
        self.n_fft = n_fft
        self.fft_bins = self.n_fft // 2 + 1

        if win_type == "hamming":
            self.win = np.hamming(self.n_fft)
        else:
            raise NotImplementedError("Invalid window type.")

        self.eps = eps

    def process(self, data):
        orig_len = len(data)

        # 1. 短于 frame_len 的音频补零, 保证至少一帧
        if len(data) < self.frame_len:
            data = np.pad(data, (0, self.frame_len - len(data)))

        # 2. 尾部补零使帧数完整, 否则不足一帧的尾部样本会被丢弃, 输出比输入短
        pad_len = (self.hop_len - (len(data) - self.frame_len) % self.hop_len) % self.hop_len
        if pad_len:
            data = np.pad(data, (0, pad_len))
        frames = librosa.util.frame(data, frame_length=self.frame_len, hop_length=self.hop_len)

        # 3. 准备输出数组（与补零后的输入等长）
        output_size = (frames.shape[1] - 1) * self.hop_len + self.frame_len
        processed_data = np.zeros(output_size)

        for i in range(frames.shape[1]):
            frame = frames[:, i]

            # --- 频域处理 ---
            win_frame = frame * self.win # 分析窗
            spectrum = np.fft.rfft(win_frame, self.n_fft)

            # 计算增益 (IMCRA 噪声估计 + OMLSA 谱增益)
            frame_psd = np.abs(spectrum) ** 2
            noise_psd = self.noise_estimator.estimate_noise(frame_psd)
            gain, _ = self.spectral_gain_estimator.compute_gain(frame_psd, noise_psd)

            # 应用增益
            processed_spectrum = spectrum * gain
            # 逆变换回时域
            processed_frame = np.fft.irfft(processed_spectrum, self.n_fft)

            # --- 关键修正：完整重叠累加 ---
            # 必须再次乘以窗函数（合成窗），以保证重叠处的平滑过渡
            processed_frame = processed_frame[:self.frame_len] * self.win

            start = i * self.hop_len
            end = start + self.frame_len
            processed_data[start:end] += processed_frame

        return processed_data[:orig_len]


def build_denoiser(args):
    """按命令行参数构造降噪器 (噪声估计器有状态, 每个文件需重新构造)"""
    noise_estimator = IMCRANoiseEstimator(n_fft=args.n_fft)
    spectral_gain_estimator = OMLSASpectralGainEstimator(n_fft=args.n_fft)
    return IMCRA_OMLSA(
        noise_estimator, spectral_gain_estimator,
        n_fft=args.n_fft, frame_len=args.frame_len,
        hop_len=args.hop_len, win_type=args.win_type,
    )


def main(args):
    # 读取 scp 文件, 每行支持三种格式:
    #   "uid 音频路径"               (与 evaluation 脚本约定一致, clean = ref_dir/uid.wav)
    #   "uid 音频路径 clean路径"      (clean 由第三列指定)
    #   "uid"                       (音频路径 = wav_dir/uid.wav)
    #   "音频路径"                   (uid = 文件名去掉 .wav)
    scp_pairs = []
    with open(args.scp, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if len(tokens) == 1 and tokens[0].endswith('.wav'):
                noisy_path = tokens[0]
                uid = os.path.splitext(os.path.basename(noisy_path))[0]
            else:
                uid = tokens[0]
                if len(tokens) >= 2:
                    noisy_path = tokens[1]
                else:
                    if not args.wav_dir:
                        raise ValueError(f"scp 行只有 uid '{uid}', 请通过 --wav_dir 指定音频目录")
                    noisy_path = os.path.join(args.wav_dir, uid + '.wav')
            if len(tokens) >= 3:
                clean_path = tokens[2]
            elif args.ref_dir:
                clean_path = os.path.join(args.ref_dir, uid + '.wav')
            else:
                clean_path = None
            scp_pairs.append((uid, noisy_path, clean_path))

    enh_folder = args.enh_dir
    os.makedirs(enh_folder, exist_ok=True)

    inf_scp_list = []
    ref_scp_list = []
    for uid, noisy_path, clean_path in tqdm(scp_pairs):
        if clean_path and not os.path.exists(clean_path):
            print(f"[WARN] clean 文件不存在, 跳过: {clean_path}")
            continue

        noisy, fs = sf.read(noisy_path, dtype='float32')
        if noisy.ndim > 1:
            noisy = noisy.mean(axis=1) # 多声道取平均

        # 噪声估计器/增益估计器跨帧有状态, 每个文件重新初始化
        enhanced = build_denoiser(args).process(noisy)

        enh_path = os.path.join(enh_folder, uid + "_enh.wav")
        inf_scp_list.append([uid, enh_path])
        sf.write(enh_path, enhanced, fs)

        if clean_path:
            ref_scp_list.append([uid, clean_path])

    # 与 infer.py 一致, 输出 inf.scp/ref.scp 供 evaluation 脚本计算指标
    with open(os.path.join(enh_folder, "inf.scp"), "w") as f:
        for uid, audio_path in inf_scp_list:
            f.write(f"{uid} {audio_path}\n")

    if ref_scp_list:
        with open(os.path.join(enh_folder, "ref.scp"), "w") as f:
            for uid, audio_path in ref_scp_list:
                f.write(f"{uid} {audio_path}\n")
    print(f"Enhanced wavs and inf.scp/ref.scp saved to {enh_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMCRA + OMLSA 降噪, 批量推理 scp 文件")
    parser.add_argument('--scp', required=True, help='输入 scp 文件, 每行 "uid 音频路径" 或仅 "uid"')
    parser.add_argument('--wav_dir', default='', help='noisy 音频目录, scp 行仅含 uid 时拼接 wav_dir/uid.wav')
    parser.add_argument('--ref_dir', default='', help='clean 音频目录, 用于生成 ref.scp (ref_dir/uid.wav)')
    parser.add_argument('--enh_dir', default='runs/imcra_omlsa/enhanced', help='增强音频输出目录')
    parser.add_argument('--n_fft', type=int, default=256)
    parser.add_argument('--frame_len', type=int, default=256)
    parser.add_argument('--hop_len', type=int, default=128)
    parser.add_argument('--win_type', default='hamming')

    args = parser.parse_args()
    main(args)
