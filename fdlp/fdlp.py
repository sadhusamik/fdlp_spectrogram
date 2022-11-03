import sys

import numpy as np
from random import randrange
from typeguard import check_argument_types
from scipy.fftpack import dct, idct, dst, idst
import pickle as pkl
from scipy.interpolate import interp1d
import librosa
import logging


class FDLP:
    def __init__(self,
                 n_filters: int = 80,
                 coeff_num: int = 80,
                 coeff_range: str = '1,80',
                 order: int = 80,
                 fduration: float = 1.5,
                 frate: int = 100,
                 overlap_fraction: float = 0.5,
                 lifter_file: str = None,
                 lfr: float = 33,  # only used when return_mvector = True
                 return_mvector: bool = False,
                 complex_mvectors: bool = True,
                 interpolate: bool = True,
                 return_phase: bool = False,
                 no_window: bool = False,
                 use_gl: bool = False,
                 fbank_config: str = '1,1,2.5',
                 normalize_uttwise_variance: bool = False,
                 spectral_substraction_signal: np.array = None,
                 spectral_substraction_vector: np.array = None,
                 clean_spectral_substraction_vector: np.array = None,
                 online_normalize: bool = False,
                 feature_batch: int = None,
                 srate: int = 16000):
        assert check_argument_types()

        self.use_gl = use_gl
        self.fbank_config = [float(x) for x in fbank_config.split(',')]
        self.n_filters = n_filters
        self.coeff_num = coeff_num
        coeff_range = coeff_range.split(',')
        self.lowpass = int(coeff_range[0])
        self.highpass = int(coeff_range[1])
        self.order = order
        self.fduration = fduration
        self.frate = frate
        self.return_mvector = return_mvector
        self.srate = srate
        self.overlap_fraction = 1 - overlap_fraction
        self.no_window = no_window
        self.complex_mvectors = complex_mvectors
        self.interpolate = interpolate
        self.return_phase = return_phase
        self.online_normalize = online_normalize
        if return_mvector:
            self.lfr = lfr
        else:
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
        if self.complex_mvectors:
            self.fbank = self.initialize_filterbank(self.n_filters, int(self.fduration * self.srate), self.srate,
                                                    om_w=self.fbank_config[0],
                                                    alp=self.fbank_config[1], fixed=1, bet=self.fbank_config[2],
                                                    warp_fact=1, make_symmetric=True)
        else:
            self.fbank = self.initialize_filterbank(self.n_filters, int(2 * self.fduration * self.srate), self.srate,
                                                    om_w=self.fbank_config[0],
                                                    alp=self.fbank_config[1], fixed=1, bet=self.fbank_config[2],
                                                    warp_fact=1)
        if lifter_file is not None:
            self.lifter = pkl.load(open(lifter_file, 'rb'))
        else:
            self.lifter = np.ones(coeff_num)
        self.normalize_uttwise_variance = normalize_uttwise_variance

        # These can be changed inside the forward function
        self.t_samples = None
        self.modspec_phase = None
        self.modspec_magnitude = None
        self.input_dtype = None

        if spectral_substraction_signal is not None:
            if spectral_substraction_signal.shape[0] < self.srate * self.fduration:
                # append zeros
                self.spectral_substraction_vector = np.log(np.fft.fft(np.concatenate((spectral_substraction_signal,
                                                                                      np.zeros(
                                                                                          int(self.srate * self.fduration) -
                                                                                          spectral_substraction_signal.shape[
                                                                                              0])))))
            else:
                self.spectral_substraction_vector = np.log(np.fft.fft(
                    spectral_substraction_signal[0:self.srate * self.fduration]))
        else:
            self.spectral_substraction_vector = None

        if spectral_substraction_vector is not None:
            self.spectral_substraction_vector = spectral_substraction_vector

        self.clean_spectral_substraction_vector = clean_spectral_substraction_vector
        self.reconstructed_speech = None
        self.reconstructed_speech_chunk = None
        self.logmag = None
        self.ph = None
        self.feature_batch = feature_batch

    def initialize_filterbank(self, nfilters, nfft, srate, om_w=1.0, alp=1.0, fixed=1, bet=2.5, warp_fact=1,
                              make_symmetric=False):
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
        if make_symmetric:
            return np.concatenate((filts[:, :-1], np.flip(filts, axis=1)), axis=1)
        else:
            return filts

    def __warp_func_bark(self, x, warp_fact=1):
        return 6 * np.arcsinh((x / warp_fact) / 600)

    def compute_autocorr(self, input):
        """

        :param input: Array (Batch x time_frames x dimension)
        :return: Array (Autocorrelation coefficients)
        """
        r = np.fft.ifft(np.fft.fft(input) * np.conj(np.fft.fft(input)))
        return r

    def levinson_durbin(self, R, p):
        """
        Levinson Durbin recursion to compute LPC coefficients

        :param R: autocorrelation coefficients - Tensor (batch x num_frames x n_filters  x autocorr)
        :param p: lpc model order - int
        :return: Tensor (batch x n_filters x lpc_coeff), Tensor (batch x num_frames x n_filters)

        """
        num_batch = R.shape[0]
        num_frames = R.shape[1]
        n_filters = R.shape[2]

        k = np.zeros((num_batch, num_frames, n_filters, p), dtype=R.dtype)
        alphs = np.zeros((num_batch, num_frames, n_filters, p, p), dtype=R.dtype)
        errs = np.zeros((num_batch, num_frames, n_filters, p + 1), dtype=R.dtype)
        errs[:, :, :, 0] = R[:, :, :, 0]
        for i in range(1, p + 1):
            if i == 1:
                k[:, :, :, i - 1] = R[:, :, :, i] / errs[:, :, :, i - 1]
            else:
                k[:, :, :, i - 1] = (R[:, :, :, i] - np.sum(
                    alphs[:, :, :, 0:i - 1, i - 2] * np.flip(R[:, :, :, 1:i], [3]), axis=3)) / errs[:, :, :, i - 1]
            alphs[:, :, :, i - 1, i - 1] = k[:, :, :, i - 1]
            if i > 1:
                for j in range(1, i):
                    alphs[:, :, :, j - 1, i - 1] = alphs[:, :, :, j - 1, i - 2] - k[:, :, :, i - 1] * np.conj(
                        alphs[:, :, :,
                        i - j - 1,
                        i - 2])
            errs[:, :, :, i] = (1 - np.abs(k[:, :, :, i - 1]) ** 2) * errs[:, :, :, i - 1]

        return np.concatenate(
            (np.ones((num_batch, num_frames, n_filters, 1), dtype=R.dtype), -alphs[:, :, :, :, p - 1]), axis=3), errs[:,
                                                                                                                 :, :,
                                                                                                                 -1]

    def compute_lpc(self, input, order):

        """
        Compute LPC coefficients
        :param input: Array (batch x num_frames x n_filters x frame_dim)
        :return: Array (batch x num_frames x n_filters x lpc_coeff), Array (batch x num_frames x n_filters)
        """
        R = self.compute_autocorr(input)
        lpc_coeff, gain = self.levinson_durbin(R, p=order)
        return lpc_coeff, gain

    def modspec_2_fdlpresponse(self, signal):

        """
        FDLP response from modulation spectrum
        :param signal: Array (batch x num_frames x n_filters x frame_dim)
        :return: Array (batch x num_frames x n_filters x int(self.fduration * self.frate))
        """
        if self.complex_mvectors:
            signal = np.fft.fft(signal, 1 * int(
                self.fduration * self.frate))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        else:
            signal = np.fft.fft(signal, 2 * int(
                self.fduration * self.frate))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        return np.abs(np.exp(signal))

    def mask_n_lifter(self, signal):
        """

        :param signal: Array (batch x num_frames x n_filters x num_modspec)
        :return: Array (batch x num_frames x n_filters x num_modspec)
        """
        signal = signal * self.mask
        signal = signal * self.lifter

        return signal

    def compute_modspec_from_lpc(self, gain, lpc_coeff, lim):
        """
        :param gain: Array (batch x num_frames x n_filters)
        :param lpc_coeff: Array (batch x num_frames x n_filters x lpc_num)
        :param lim: int
        :return: Array (batch x num_frames x n_filters x num_modspec),
        """

        num_batch = lpc_coeff.shape[0]
        num_frames = lpc_coeff.shape[1]
        n_filters = lpc_coeff.shape[2]
        lpc_coeff[:, :, :, 1:] = -lpc_coeff[:, :, :, 1:]
        lpc_cep = np.zeros((num_batch, num_frames, n_filters, lim), dtype=lpc_coeff.dtype)
        lpc_cep[:, :, :, 0] = np.log(np.sqrt(gain))
        lpc_cep[:, :, :, 1] = lpc_coeff[:, :, :, 1]
        if lpc_coeff.shape[3] < lim:
            lpc_coeff = np.concatenate(
                [lpc_coeff, np.zeros((num_batch, num_frames, n_filters, int(lim - lpc_coeff.shape[3] + 1)))],
                axis=3)
        for n in range(2, lim):
            a = np.arange(1, n) / n
            b = np.flip(lpc_coeff[:, :, :, 1:n], axis=[3])
            c = lpc_cep[:, :, :, 1:n]
            acc = np.sum(a * b * c, axis=3)
            lpc_cep[:, :, :, n] = acc + lpc_coeff[:, :, :, n]
        return lpc_cep

    def get_frames(self, signal, no_window=False, reflect=True):
        """Divide speech signal into frames.

                Args:
                    signal: (Batch, Nsamples) or (Batch, Nsample)
                Returns:
                    output: (Batch, Frame num, Frame dimension) or (Batch, Frame num, Frame dimension)
                """

        flength_samples = int(self.srate * self.fduration)
        frate_samples = int(self.srate / self.lfr)

        if self.feature_batch is not None:
            # Reshape to have longer utterances, helps in feature extraction
            signal = np.reshape(signal, (self.feature_batch, -1))

        if flength_samples % 2 == 0:
            sp_b = int(flength_samples / 2) - 1
            sp_f = int(flength_samples / 2)
            extend = int(flength_samples / 2) - 1
        else:
            sp_b = int((flength_samples - 1) / 2)
            sp_f = int((flength_samples - 1) / 2)
            extend = int((flength_samples - 1) / 2)

        if reflect:
            signal = np.pad(signal, ((0, 0), (extend, extend)), 'reflect')
        else:
            signal = np.pad(signal, ((0, 0), (extend, extend)), 'constant')

        signal_length = signal.shape[1]
        win = np.hamming(flength_samples)
        idx = sp_b
        frames = []
        while (idx + sp_f) < signal_length:
            if no_window:
                frames.append(signal[:, np.newaxis, idx - sp_b:idx + sp_f + 1])
            else:
                frames.append(signal[:, np.newaxis, idx - sp_b:idx + sp_f + 1] * win)
            idx += frate_samples

        if len(frames) == 0:
            # Just make frame
            logging.info('Sentence too short, only making on frame with given configuration..')
            sys.stdout.flush()
            frames.append(np.concatenate([signal[:, np.newaxis, idx - sp_b:idx + sp_f + 1],
                                          np.zeros((signal.shape[0], 1, flength_samples - signal.shape[1]))], axis=2))

        return np.concatenate(frames, axis=1)

    def OLA(self, modspec, t_samples, dtype):
        """
        Overlap-Add of self.fduration length spectrograms

        :param modspec:
        :param t_samples:
        :param dtype:
        :return:
        """

        num_batch = modspec.shape[0]
        num_frames = modspec.shape[1]
        feats = np.zeros((num_batch, int(np.ceil(t_samples * self.frate / self.srate)), self.n_filters), dtype=dtype)
        ptr = int(0)
        ### Overlap and Add stage
        for j in range(0, num_frames):
            if j == 0:
                if feats.shape[1] < self.cut_half:
                    feats += modspec[:, j, :self.cut_half:self.cut_half + feats.shape[1], :]
                else:
                    feats[:, ptr:ptr + self.cut_half, :] += modspec[:, j, self.cut_half:, :]

            elif j == num_frames - 1 or j == num_frames - 2:
                if modspec.shape[2] >= feats.shape[1] - ptr:
                    feats[:, ptr:, :] += modspec[:, j, :feats.shape[1] - ptr, :]
                else:
                    feats[:, ptr:ptr + self.cut, :] += modspec[:, j, :, :]
            else:
                feats[:, ptr:ptr + self.cut, :] += modspec[:, j, :, :]

            if j == 0:
                ptr = int(ptr + self.cut_overlap - self.cut_half)
            else:
                # ptr = int(ptr + self.cut_overlap + randrange(2))
                ptr = int(ptr + self.cut_overlap + 1)

        feats = np.log(np.clip(feats, a_max=None, a_min=0.0000001))

        return feats

    def spectral_substraction_preprocessing(self, frames):
        ori_len = frames.shape[-1]
        if self.spectral_substraction_vector.shape[0] > frames.shape[-1]:
            frames = np.concatenate([frames, np.zeros(
                (frames.shape[0], frames.shape[1], self.spectral_substraction_vector.shape[0] - frames.shape[-1]))],
                                    axis=-1)
        else:
            frames = frames[:, :0:self.spectral_substraction_vector.shape[0]]

        frames_fft = np.log(np.fft.fft(frames))
        frames_fft_ph = np.unwrap(np.imag(frames_fft))
        frames_fft = np.real(frames_fft) + 1j * frames_fft_ph
        if self.clean_spectral_substraction_vector is not None:
            frames_fft = np.real(np.fft.ifft(
                np.exp(frames_fft - self.spectral_substraction_vector + self.clean_spectral_substraction_vector)))
        else:
            frames_fft = np.real(np.fft.ifft(np.exp(frames_fft - self.spectral_substraction_vector)))

        return frames_fft[:, :, :ori_len]

    def acc_log_spectrum_fft_old(self, input):
        frames = self.get_frames(input, no_window=self.no_window)
        frames = np.fft.fft(frames[0])
        frames_mag = np.abs(frames)
        frames_ang = np.unwrap(np.angle(frames))
        frames_mag = np.sum(np.log(frames_mag), axis=0)
        frames_ang = np.sum(frames_ang, axis=0)
        # frames_ang = (frames_ang + np.pi) % (2 * np.pi) - np.pi
        return frames.shape[0], frames_mag, frames_ang

    def acc_log_spectrum(self, input, append_len=500000):

        input = np.concatenate([input, np.zeros(append_len - input.shape[0])])
        input = input[0:append_len]

        frames_dct = dct(input, type=2)
        frames_dst = dst(input, type=2)

        return 1, frames_dct, frames_dst

    def acc_log_spectrum_fft(self, input, append_len=500000, discont=np.pi):

        input = np.concatenate([input, np.zeros(append_len - input.shape[0])])
        input = input[0:append_len]
        frames_fft = np.log(np.fft.fft(input))

        return 1, np.real(frames_fft), np.unwrap(np.imag(frames_fft), discont=discont)

    def acc_log_spectrum_fft_frames_old(self, input, append_len=500000, discont=np.pi):

        angg = 8000 * 2 * np.pi * self.overlap_fraction * self.fduration / append_len

        input = input[None, :]
        input = self.get_frames(input, no_window=True, reflect=False)
        input = input[0]
        num_frames = input.shape[0]
        input = np.concatenate([input, np.zeros((num_frames, append_len - input.shape[1]))], axis=-1)
        input = input[:, 0:append_len]
        frames_fft = np.log(np.fft.fft(input, axis=-1))
        phase_all = np.unwrap(np.imag(frames_fft), discont=discont, axis=-1)
        cc = 0
        for i in range(phase_all.shape[0]):
            phase_all[i] = phase_all[i] * (1 + cc * angg)
            cc += 1
        return num_frames, np.sum(np.real(frames_fft), axis=0), np.sum(phase_all, axis=0)

    def acc_log_spectrum_fft_frames(self, input, append_len=500000, discont=np.pi):

        input = input[None, :]
        input = self.get_frames(input, no_window=True, reflect=False)
        input = input[0]
        num_frames = input.shape[0]
        input = np.concatenate([input, np.zeros((num_frames, append_len - input.shape[1]))], axis=-1)
        input = input[:, 0:append_len]
        frames_fft = np.log(np.fft.fft(input, axis=-1))
        phase_all = np.unwrap(np.imag(frames_fft), discont=discont, axis=-1)

        for idx, phase in enumerate(phase_all):
            phi = (phase[-1] - phase[0]) / phase.shape[0]
            x_ph = np.arange(phase.shape[0])
            y_ph = phase[0] + x_ph * phi
            ph_corrected = phase - y_ph
            phase_all[idx] = ph_corrected

        return num_frames, np.sum(np.real(frames_fft), axis=0), np.sum(phase_all, axis=0)

    def get_normalizing_vector(self, signal, fduration, overlap_fraction, append_len=1000000, discont=np.pi,
                               no_window=True, phase_max_cap=200):

        # Divide speech into windows
        overlap_fraction = 1 - overlap_fraction
        lfr = 1 / (overlap_fraction * fduration)
        flength_samples = int(self.srate * fduration)
        frate_samples = int(self.srate / lfr)

        if self.feature_batch is not None:
            # Reshape to have longer utterances, helps in feature extraction
            signal = np.reshape(signal, (self.feature_batch, -1))

        if flength_samples % 2 == 0:
            sp_b = int(flength_samples / 2) - 1
            sp_f = int(flength_samples / 2)
            extend = int(flength_samples / 2) - 1
        else:
            sp_b = int((flength_samples - 1) / 2)
            sp_f = int((flength_samples - 1) / 2)
            extend = int((flength_samples - 1) / 2)

        signal = np.pad(signal, ((0, 0), (extend, extend)), 'constant')

        signal_length = signal.shape[1]
        win = np.hamming(flength_samples)
        idx = sp_b
        frames = []
        while (idx + sp_f) < signal_length:
            if no_window:
                frames.append(signal[:, np.newaxis, idx - sp_b:idx + sp_f + 1])
            else:
                frames.append(signal[:, np.newaxis, idx - sp_b:idx + sp_f + 1] * win)
            idx += frate_samples

        if len(frames) == 0:
            # Just make frame
            logging.info('Sentence too short, only making on frame with given configuration..')
            sys.stdout.flush()
            frames.append(np.concatenate([signal[:, np.newaxis, idx - sp_b:idx + sp_f + 1],
                                          np.zeros((signal.shape[0], 1, flength_samples - signal.shape[1]))], axis=2))

        frames = np.concatenate(frames, axis=1)

        frames = np.concatenate([frames, np.zeros((frames.shape[0], frames.shape[1], append_len - frames.shape[2]))],
                                axis=-1)
        frames = frames[:, :, 0:append_len]
        frames = np.log(np.fft.fft(frames, axis=-1))
        frames = np.reshape(frames, (frames.shape[0] * frames.shape[1], -1))

        total_num_frames = frames.shape[0]
        phase = np.unwrap(np.imag(frames), discont=discont, axis=-1)
        logmag = np.real(frames)

        for idx, phs in enumerate(phase):
            phi = (phs[-1] - phs[0]) / phs.shape[0]
            x_ph = np.arange(phs.shape[0])
            y_ph = phase[0] + x_ph * phi
            ph_corrected = phs - y_ph
            phase[idx] = ph_corrected

        phase = np.sum(phase, axis=0) / total_num_frames
        logmag = np.sum(logmag, axis=0) / total_num_frames

        ## Adjust the phase
        #phi = (phase[-1] - phase[0]) / phase.shape[0]
        #x_ph = np.arange(phase.shape[0])
        #y_ph = phase[0] + x_ph * phi
        #ph_corrected = y_ph - phase
        #ph_corrected = ph_corrected * phase_max_cap / np.max(ph_corrected)

        ssv = logmag + 1j * phase

        return ssv

    def acc_log_spectrum_fft_frames_separated(self, input, append_len=500000, discont=np.pi):

        input = input[None, :]
        input = self.get_frames(input, no_window=True, reflect=False)
        input = input[0]
        num_frames = input.shape[0]
        input = np.concatenate([input, np.zeros((num_frames, append_len - input.shape[1]))], axis=-1)
        input = input[:, 0:append_len]
        frames_fft = np.log(np.fft.fft(input, axis=-1))

        return num_frames, np.real(frames_fft), np.unwrap(np.imag(frames_fft), discont=discont, axis=-1)

    def compute_spectrogram(self, input, ilens=None):
        """Main function that computes FDLp spectrogram.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, n_filters) or (Batch, Frames, n_filters)

        """
        t_samples = input.shape[1]
        self.t_samples = t_samples
        self.input_dtype = input.dtype
        num_batch = input.shape[0]

        if self.online_normalize:
            # Compute ssv on the fly
            self.spectral_substraction_vector = self.get_normalizing_vector(input, fduration=25, overlap_fraction=0.98,
                                                                            append_len=1000000)

        # First divide the signal into frames
        frames = self.get_frames(input, no_window=self.no_window)
        num_frames = frames.shape[1]

        if self.spectral_substraction_vector is not None:
            frames = self.spectral_substraction_preprocessing(frames)
            self.reconstructed_speech_chunk = frames
        # Compute DCT/FFT (olens remains the same)
        if self.complex_mvectors:
            frames = np.fft.ifft(frames) * frames.shape[1]  # [:, :, 0:int(frames.shape[2]/2)]
        else:
            frames = dct(frames) / np.sqrt(2 * int(self.srate * self.fduration))

        fbank = self.fbank

        frames = np.tile(frames[:, :, np.newaxis, :], (1, 1, self.n_filters, 1))
        frames = frames * fbank[:, 0:-1]  # batch x num_frames x n_filters x frame_dim of 1.5 secs

        frames, gain = self.compute_lpc(frames, self.order)  # batch x num_frames x n_filters x lpc_coeff
        frames = self.compute_modspec_from_lpc(gain, frames,
                                               self.coeff_num)  # batch x num_frames x n_filters x num_modspec
        modspec = frames

        if self.return_mvector:
            if self.complex_mvectors:
                if self.return_phase:
                    modspec = np.concatenate((np.abs(modspec), np.angle(modspec)),
                                             axis=3)  # Return magnitude and phase of modulation spectrum
                else:
                    modspec = np.abs(modspec)  # Return only magnitude spectrum
            if self.interpolate and self.lfr != self.frate:
                # We have to interpolate using splines features to frame rate
                modspec = modspec.reshape(
                    (modspec.shape[0], modspec.shape[1], -1))  # batch x num_frames x n_filters * num_modspec
                modspec = modspec.transpose((0, 2, 1))  # batch x n_filters * num_modspec x num_frames
                x = np.arange(num_frames)
                f = interp1d(x, modspec, axis=2, kind='cubic')
                num_frames_interpolated = int(num_frames * self.frate / self.lfr)
                x_interpolated = np.linspace(0, num_frames - 1, num_frames_interpolated)
                modspec = f(x_interpolated)
                modspec = modspec.transpose((0, 2, 1))  # batch  x num_frames_interpolated x n_filters * num_modspec
            else:
                modspec = modspec.reshape(
                    (modspec.shape[0], modspec.shape[1], -1))  # batch x num_frames x n_filters * num_modspec

        else:
            # if self.complex_mvectors:
            # modspec = modspec[:, :, :, ::2]
            modspec = self.mask_n_lifter(modspec)  # (batch x num_frames x n_filters x num_modspec)
            modspec = self.modspec_2_fdlpresponse(
                modspec)  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
            modspec = modspec[:, :, :, 0:self.cut] * np.hanning(self.cut) / np.hamming(self.cut)
            modspec = np.transpose(modspec,
                                   (0, 1, 3, 2))  # (batch x num_frames x int(self.fduration * self.frate) x n_filters)

            # OVERLAP AND ADD
            if self.feature_batch is not None:
                modspec = self.OLA(modspec=modspec, t_samples=int(t_samples * num_batch / self.feature_batch),
                                   dtype=input.dtype)
            else:
                modspec = self.OLA(modspec=modspec, t_samples=t_samples, dtype=input.dtype)

        if self.feature_batch is not None:
            # Might not be equally divisible, deal with that
            modspec_size = modspec.shape[0] * modspec.shape[1] * modspec.shape[2]
            div_req = num_batch * self.n_filters
            div_reminder = modspec_size % div_req
            if div_reminder != 0:
                modspec = modspec.flatten()
                if div_reminder < int(div_req / 2):
                    modspec = modspec[:-div_reminder]
                else:
                    modspec = np.concatenate([modspec, np.zeros(div_reminder)])

            modspec = np.reshape(modspec, (num_batch, -1, self.n_filters))
        if ilens is not None:
            olens = np.round(ilens * self.frate / self.srate)
        else:
            olens = None

        return modspec, olens

    def modspec_2_spectrum(self, modspec):
        modspec = self.mask_n_lifter(modspec)  # (batch x num_frames x n_filters x num_modspec)
        modspec = self.modspec_2_fdlpresponse(
            modspec)  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        modspec = modspec[:, :, :, 0:self.cut] * np.hanning(self.cut) / np.hamming(self.cut)
        modspec = np.transpose(modspec,
                               (0, 1, 3, 2))  # (batch x num_frames x int(self.fduration * self.frate) x n_filters)

        # OVERLAP AND ADD
        modspec = self.OLA(modspec=modspec, t_samples=self.t_samples, dtype=self.input_dtype)

        return modspec

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
        nsamples = input.shape[1]
        if len(input.shape) == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose((0, 2, 1)).reshape((-1, input.shape[1]))
        else:
            multi_channel = False

        if self.normalize_uttwise_variance and not multi_channel:
            input /= np.tile(np.var(input, axis=1), (nsamples, 1)).T

        # Compute FDLP spectrogram
        output, olens = self.compute_spectrogram(input, ilens)

        if multi_channel:
            # output: (Batch * Channel, Frames, n_filters)
            # -> (Batch, Frame, Channel, n_filters)
            output = output.reshape((bs, -1, output.shape[1], output.shape[2])).transpose(0, 2, 1, 3)

        return output, olens
