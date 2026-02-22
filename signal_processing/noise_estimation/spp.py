'''
Author: Ryuk
Date: 2026-02-17 16:00:39
LastEditors: Ryuk
LastEditTime: 2026-02-17 16:03:16
Description: First create
'''

import numpy as np

from ..base import BaseNoiseEstimator

    
class SPPNoiseEstimator(BaseNoiseEstimator):

    """
    T. Gerkmann and R. C. Hendriks, “Unbiased MMSE-based noise power
    estimation with low complexity and low tracking delay,” IEEE
    Transactions on Audio, Speech, and Language Processing, vol. 20, no. 4,
    pp. 1383–1393, May 2012.
    """

    def __init__(self, n_fft,
                 fixed_smooth=0.8,
                 prob_smooth=0.9,
                 prior=0.5,
                 snr_opt_db=1,
                 num_frames_init=0):

        self.n_fft = n_fft
        self.fft_bins = n_fft // 2 + 1

        # fixed smoothing constant for smoothing the noise periodogram
        self._fixed_smooth = fixed_smooth

        # fixed smoothing constant for smoothing the SPP
        self._prob_smooth = prob_smooth

        # prior probability for speech presence (P(H1) in [1])
        self._prior = prior

        # fixed SNR (\xi_opt in [1])
        self._snr_opt_lin = 10.**(snr_opt_db/10.)

        # number of frames used for initialization
        self._num_frames_init = num_frames_init

        # internal states
        self._v_old_psd = np.zeros(self.fft_bins)
        self._v_smooth_prob = np.zeros(self.fft_bins)
        self._inv_glr_factor = (1 - prior)/prior*(1. + self._snr_opt_lin)
        self._inv_glr_exp_factor = self._snr_opt_lin/(1. + self._snr_opt_lin)
        self._num_frames_processed = 0

    def estimate_noise(self, v_noisy_per, v_spp_in=None):
        if v_spp_in is None:
            if self._num_frames_processed < self._num_frames_init:
                # average first frames to obtain first noise PSD estimate
                v_noise_psd = self._v_old_psd + v_noisy_per / self._num_frames_init

                self._v_old_psd = v_noise_psd

                # increment frame counter
                self._num_frames_processed += 1

                v_spp = np.zeros_like(self._v_old_psd) # SPP considered 0 at the beginning

                return v_noisy_per, v_spp
            else:
                # compute inverse GLR
                v_inv_glr = self._inv_glr_factor * \
                    np.exp(-v_noisy_per / (self._v_old_psd + 1e-8) * self._inv_glr_exp_factor)

                # compute SPP (corresponds to line 2 in Algorithm 1, [1])
                v_spp = 1. / (1. + v_inv_glr)

                # stuck protection (corresponds to line 3 and 4 in Algorithm 1,
                # [1])
                self._v_smooth_prob = (1 - self._prob_smooth) * v_spp + \
                    self._prob_smooth * self._v_smooth_prob
                v_mask = self._v_smooth_prob > 0.99
                v_spp[v_mask] = np.minimum(v_spp[v_mask], 0.99)

                # estimate noise periodogram (corresponds to line 5 in Algorithm
                # 1, [1])
                v_noise_per = (1. - v_spp) * v_noisy_per + \
                    v_spp * self._v_old_psd
                # corresponds to line 6 in Algorithm 1, [1]
                v_noise_psd = (1. - self._fixed_smooth) * v_noise_per + \
                    self._fixed_smooth * self._v_old_psd

            # update old noise PSD estimate
            self._v_old_psd = v_noise_psd

            return v_noise_psd, v_spp
        else:
            # estimate noise periodogram (corresponds to line 5 in Algorithm
            # 1, [1])
            #TODO: better use formula of Wang with alpha combined with mask
            v_noise_per = (1. - v_spp_in) * v_noisy_per + \
                v_spp_in * self._v_old_psd
            # corresponds to line 6 in Algorithm 1, [1]
            v_noise_psd = (1. - self._fixed_smooth) * v_noise_per + \
                self._fixed_smooth * self._v_old_psd
        return v_noise_psd
