import sys

import numpy as np
from random import randrange
from typeguard import check_argument_types
import scipy.fftpack as freqAnalysis
import pickle as pkl


class fdlp:
    def __init__(self,
                 n_filters: int = 80,
                 coeff_num: int = 100,
                 coeff_range: str = '1,100',
                 order: int = 150,
                 fduration: float = 1.5,
                 frate: int = 100,
                 overlap_fraction: float = 0.25,
                 lifter_file: str = None,
                 srate: int = 16000):
        assert check_argument_types()

        self.n_filters = n_filters
        self.coeff_num = coeff_num
        coeff_range = coeff_range.split(',')
        self.lowpass = int(coeff_range[0])
        self.highpass = int(coeff_range[1])
        self.order = order
        self.fduration = fduration
        self.frate = frate
        self.srate = srate
        self.overlap_fraction = 1 - overlap_fraction
        self.lfr = 1 / (self.overlap_fraction * self.fduration)
        mask = []
        for i in range(coeff_num):
            if i >= self.lowpass and i <= self.highpass:
                mask.append(1)
            else:
                mask.append(0)
        mask = np.asarray(mask)
        self.mask = mask
        self.cut = int(np.round(self.fduration * self.frate))
        self.cut_half = int(np.round(self.fduration * self.frate / 2))
        self.cut_overlap = int(np.round(self.fduration * self.frate * self.overlap_fraction))
        self.fbank = self.initialize_filterbank(self.n_filters, int(2 * self.fduration * self.srate), self.srate,
                                                om_w=1,
                                                alp=1, fixed=1, bet=2.5, warp_fact=1)
        if lifter_file is not None:
            self.lifter = pkl.load(open(lifter_file, 'rb'))
        else:
            self.lifter = np.ones(coeff_num)

    def initialize_filterbank(self, nfilters, nfft, srate, om_w=1, alp=1, fixed=1, bet=2.5, warp_fact=1):
        f_max = srate / 2
        warped_max = self.__warp_func_bark(f_max, warp_fact)
        fwarped_cf = np.linspace(0, warped_max, nfilters)
        f_linear = np.linspace(0, f_max, int(np.floor(nfft / 2 + 1)))
        f_warped = self.__warp_func_bark(f_linear, warp_fact)
        filts = np.zeros((nfilters, int(np.floor(nfft / 2 + 1))))
        alp_c = alp
        for i in range(nfilters):
            fc = fwarped_cf[i]
            if fixed == 1:
                alp = alp_c
            else:
                alp = alp_c * np.exp(-0.1 * fc)
            for j, fw in enumerate(f_warped):
                if fw - fc <= -om_w / 2:
                    filts[i, j] = np.power(10, alp * (fw - fc + om_w / 2))
                elif fw - fc > -om_w / 2 and fw - fc < om_w / 2:
                    filts[i, j] = 1
                else:
                    filts[i, j] = np.power(10, -bet * (fw - fc - om_w / 2))

        return filts

    def __warp_func_bark(self, x, warp_fact=1):
        return 6 * np.arcsinh((x / warp_fact) / 600)

    def compute_autocorr(self, input):
        """

        :param input: Array (Batch x time_frames x dimension)
        :return: Array (Autocorrelation coefficients)
        """
        r = np.real(np.fft.ifft(np.fft.fft(input) * np.conj(np.fft.fft(input))))
        return r

    def levinson_durbin(self, R, p):
        """
        Levinson Durbin recursion to compute LPC coefficients

        :param R: autocorrelation coefficients - Tensor (batch x n_filters x autocorr)
        :param p: lpc model order - int
        :return: Tensor (batch x n_filters x lpc_coeff), Tensor (batch x n_filters)

        """
        num_batch = R.shape[0]
        n_filters = R.shape[1]
        k = np.zeros((num_batch, n_filters, p), dtype=R.dtype)
        alphs = np.zeros((num_batch, n_filters, p, p), dtype=R.dtype)
        errs = np.zeros((num_batch, n_filters, p + 1), dtype=R.dtype)
        errs[:, :, 0] = R[:, :, 0]
        for i in range(1, p + 1):
            if i == 1:
                k[:, :, i - 1] = R[:, :, i] / errs[:, :, i - 1]
            else:
                k[:, :, i - 1] = (R[:, :, i] - np.sum(
                    alphs[:, :, 0:i - 1, i - 2] * np.flip(R[:, :, 1:i], [2]), axis=2)) / errs[:, :, i - 1]
            alphs[:, :, i - 1, i - 1] = k[:, :, i - 1]
            if i > 1:
                for j in range(1, i):
                    alphs[:, :, j - 1, i - 1] = alphs[:, :, j - 1, i - 2] - k[:, :, i - 1] * alphs[:, :, i - j - 1,
                                                                                             i - 2]
            errs[:, :, i] = (1 - k[:, :, i - 1] ** 2) * errs[:, :, i - 1]

        return np.concatenate((np.ones((num_batch, n_filters, 1), dtype=R.dtype), -alphs[:, :, :, p - 1]),
                              axis=2), errs[
                                       :, :,
                                       -1]

    def compute_lpc_fast(self, input, order):

        """

        :param input: Array (batch x n_filters x frame_dim)
        :return: Array (batch x n_filters x lpc_coeff), Array (batch x n_filters)
        """
        R = self.compute_autocorr(input)
        lpc_coeff, gain = self.levinson_durbin(R, p=order)
        return lpc_coeff, gain

    def modspec_2_fdlpresponse(self, signal):
        """
        :param signal: Array (batch x n_filters x frame_dim)
        :return: Array (batch x n_filters x lpc_coeff), Array (batch x n_filters)
        """
        signal = np.fft.fft(signal, 2 * int(
            self.fduration * self.frate))  # (batch x n_filters x int(self.fduration * self.frate))
        return np.abs(np.exp(signal))

    def mask_n_lifter(self, signal):
        """

        :param signal: Array (batch x n_filters x num_modspec)
        :return:
        """
        signal = signal * self.mask
        signal = signal * self.lifter

        return signal

    def compute_modspec_from_lpc_fast(self, gain, lpc_coeff, lim):
        """
        :param gain: Array (batch x n_filters)
        :param lpc_coeff: Array (batch x n_filters x lpc_num)
        :param lim: int
        :return: Array (batch x n_filters x num_modspec),
        """

        num_batch = lpc_coeff.shape[0]
        n_filters = lpc_coeff.shape[1]
        lpc_coeff[:, :, 1:] = -lpc_coeff[:, :, 1:]
        lpc_cep = np.zeros((num_batch, n_filters, lim))
        lpc_cep[:, :, 0] = np.log(np.sqrt(gain))
        lpc_cep[:, :, 1] = lpc_coeff[:, :, 1]
        if lpc_coeff.shape[2] < lim:
            lpc_coeff = np.concatenate([lpc_coeff, np.zeros((num_batch, n_filters, int(lim - lpc_coeff.shape[2] + 1)))],
                                       axis=2)
        for n in range(2, lim):
            a = np.arange(1, n) / n
            b = np.flip(lpc_coeff[:, :, 1:n], axis=[2])
            c = lpc_cep[:, :, 1:n]
            acc = np.sum(a * b * c, axis=2)
            lpc_cep[:, :, n] = acc + lpc_coeff[:, :, n]
        return lpc_cep

    def get_frames(self, signal):
        """Divide speech signal into frames.

                Args:
                    signal: (Batch, Nsamples) or (Batch, Nsample)
                Returns:
                    output: (Batch, Frame num, Frame dimension) or (Batch, Frame num, Frame dimension)
                """

        flength_samples = int(self.srate * self.fduration)
        frate_samples = int(self.srate / self.lfr)

        if flength_samples % 2 == 0:
            sp_b = int(flength_samples / 2) - 1
            sp_f = int(flength_samples / 2)
            extend = int(flength_samples / 2) - 1
        else:
            sp_b = int((flength_samples - 1) / 2)
            sp_f = int((flength_samples - 1) / 2)
            extend = int((flength_samples - 1) / 2)

        signal = np.pad(signal, ((0, 0), (extend, extend)), 'reflect')
        signal_length = signal.shape[1]
        win = np.hamming(flength_samples)
        idx = sp_b
        frames = []
        while (idx + sp_f) < signal_length:
            frames.append(signal[:, np.newaxis, idx - sp_b:idx + sp_f + 1] * win)
            idx += frate_samples

        frames = np.concatenate(frames, axis=1)

        return frames

    def compute_spectrogram(self, input, ilens=None):
        """Main function that computes FDLp spectrogram.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, n_filters) or (Batch, Frames, n_filters)

        """
        t_samples = input.shape[1]
        num_batch = input.shape[0]

        # First divide the signal into frames
        frames = self.get_frames(input)
        num_frames = frames.shape[1]

        # Compute DCT (olens remains the same)
        frames = freqAnalysis.dct(frames) / np.sqrt(2 * int(self.srate * self.fduration))

        # Main loop to compute features
        feats = np.zeros(
            (num_batch, int(np.ceil(t_samples * self.frate / self.srate)), self.n_filters))

        fbank = self.fbank

        ptr = int(0)
        frames = np.tile(frames[:, :, np.newaxis, :], (1, 1, self.n_filters, 1))  # TODO what is this function for numpy

        band_dct_all = frames * fbank[:, 0:-1]  # batch x num_frames x n_filters x frame_dim of 1.5 secs

        for j in range(0, num_frames):
            band_dct = band_dct_all[:, j, :, :]  # batch x n_filters x frame_dim
            lpc_coeff, gain = self.compute_lpc_fast(band_dct, self.order)
            modspec = self.compute_modspec_from_lpc_fast(gain, lpc_coeff,
                                                         self.coeff_num)  # batch x n_filters x frame_dim
            modspec = self.mask_n_lifter(modspec)  # (batch x n_filters x num_modspec)
            modspec = self.modspec_2_fdlpresponse(modspec)  # (batch x n_filters x int(self.fduration * self.frate))
            modspec = modspec[:, :, 0:self.cut] * np.hanning(self.cut) / np.hamming(self.cut)
            modspec = np.transpose(modspec, (0, 2, 1))  # (batch x int(self.fduration * self.frate) x n_filters)

            if j == 0:
                if feats.shape[1] < self.cut_half:
                    feats += modspec[:, self.cut_half:self.cut_half + feats.shape[1], :]
                else:
                    feats[:, ptr:ptr + self.cut_half, :] += modspec[:, self.cut_half:, :]

            elif j == num_frames - 1 or j == num_frames - 2:
                if modspec.shape[1] >= feats.shape[1] - ptr:
                    feats[:, ptr:, :] += modspec[:, :feats.shape[1] - ptr, :]
                else:
                    feats[:, ptr:ptr + self.cut, :] += modspec
            else:
                feats[:, ptr:ptr + self.cut, :] += modspec

            if j == 0:
                ptr = int(ptr + self.cut_overlap - self.cut_half)
            else:
                ptr = int(ptr + self.cut_overlap + randrange(2))

        feats = np.log(np.clip(feats, a_max=None, a_min=0.00000000000001))

        if ilens is not None:
            olens = np.round(ilens * self.frate / self.srate)

        return feats, olens

    def extract_feats(self, input, ilens=None):
        """Compute FDLP-Spectrogram.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, n_filters) or (Batch, Frames, Channels, n_filters)

        """

        # Check if data is multichannel or single-channel
        bs = input.shape[0]  # batch size
        if len(input.shape) == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # Compute FDLP spectrogram
        output, olens = self.compute_spectrogram(input, ilens)

        if multi_channel:
            # output: (Batch * Channel, Frames, n_filters)
            # -> (Batch, Frame, Channel, n_filters)
            output = output.view(bs, -1, output.shape[1], output.shape[2]).transpose(
                1, 2
            )

        return output, olens
