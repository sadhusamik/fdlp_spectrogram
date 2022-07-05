"""
Utility functions for feature computation

Author: Samik Sadhu
"""

import numpy as np
import scipy.linalg as lpc_solve
import subprocess
import os
import sys
from scipy.io.wavfile import read


def get_kaldi_ark(feat_dict, outfile, kaldi_cmd='copy-feats'):
    with open(outfile + '.txt', 'w+') as file:
        for key, feat in feat_dict.items():
            np.savetxt(file, feat, fmt='%.3f', header=key + ' [', footer=' ]', comments='')
    cmd = kaldi_cmd + ' ark,t:' + outfile + '.txt' + ' ark,scp:' + outfile + '.ark,' + outfile + '.scp'
    subprocess.run(cmd, shell=True)
    os.remove(outfile + '.txt')


def add_noise_to_wav(sig, noise, snr):
    rand_num = int(np.floor(np.random.rand() * (len(noise) - len(sig))))
    ns = noise[rand_num:rand_num + len(sig)]
    E_s = np.mean(sig ** 2)
    E_n = np.mean(ns ** 2)
    alp = np.sqrt(E_s / (E_n * (10 ** (snr / 10))))

    return sig + alp * ns


def load_noise(noise_type):
    noise_file = "noises/" + noise_type + ".wav"

    if os.path.isfile(noise_file):

        sr, noise = read(noise_file)
    else:
        print("Noise file " + noise_file + " not found!")
        os.exit(1)

    return noise  # / np.power(2, 15)


def add_agwn(sig, noise, snr):
    P_sig = np.sum(sig ** 2) / sig.size
    P_noise = np.sum(noise ** 2) / noise.size

    if sig.size != noise.size:
        print('Signal and Noise dimension are not the same, not adding noise!')

        sig_mod = sig
    else:

        k = np.sqrt(P_sig / (P_noise * np.power(10, (snr / 10))))
        sig_mod = sig + k * noise

    return sig_mod


def dict2Ark(feat_dict, outfile, kaldi_cmd):
    with open(outfile + '.txt', 'w+') as file:
        for key, feat in feat_dict.items():
            np.savetxt(file, feat, fmt='%.3f', header=key + ' [', footer=' ]', comments='')
    cmd = kaldi_cmd + ' ark,t:' + outfile + '.txt' + ' ark,scp:' + outfile + '.ark,' + outfile + '.scp'
    subprocess.run(cmd, shell=True)
    os.remove(outfile + '.txt')


def ark2Dict(ark, dim, kaldi_cmd):
    cmd = kaldi_cmd + ' ark:' + ark + ' ark,t:-'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    x = proc.stdout.decode('utf-8')
    feat_count = 0
    start = 0
    feats = np.empty((0, dim))
    all_feats = {}
    fcount = 0;
    for line in x.splitlines():

        line = line.strip().split()
        if len(line) >= 1:
            if line[-1] == '[':
                start = 1
                feat_count += 1  # Starting of a feature
                uttname = line[0]
                feats = np.empty((0, dim))
                fcount += 1;
            if start == 1 and line[-1] != '[':
                if line[-1] == ']':
                    line = line[0:-1]
                    x = np.array(line).astype(np.float)
                    x = np.reshape(x, (1, len(x)))

                    feats = np.concatenate((feats, x), axis=0)
                    all_feats[uttname] = feats
                    # Refresh everything
                    start = 0
                    feats = np.empty((0, dim))
                else:
                    x = np.array(line).astype(np.float)
                    x = np.reshape(x, (1, len(x)))
                    feats = np.concatenate((feats, x), axis=0)
    print('%s: Tranfered %d utterances from ark to dict' % (sys.argv[0], fcount))
    return all_feats


def addReverb(sig, reverb):
    out = np.convolve(sig, reverb)
    xxc = np.correlate(sig, out, 'valid')
    indM = len(xxc) - np.argmax(xxc)
    out = out[indM:indM + len(sig)]
    return out

def addReverb_nodistortion(sig, reverb):
    out = np.convolve(sig, reverb, mode='full')
    xxc = np.correlate(sig, out, 'valid')
    indM = len(xxc) - np.argmax(xxc)
    return out, indM



def getFrames(signal, srate, frate, flength, window):
    '''Generator of overlapping frames

    Args:
        signal (numpy.ndarray): Audio signal.
        srate (float): Sampling rate of the signal.
        frate (float): Frame rate in Hz.
        flength (float): Frame length in second.
        window (function): Window function (see numpy.hamming for instance).

    Yields:
        frame (numpy.ndarray): frame of length ``flength`` every ``frate``
            second.

    '''

    flength_samples = int(srate * flength)
    frate_samples = int(srate / frate)

    if flength_samples % 2 == 0:
        sp_b = int(flength_samples / 2) - 1
        sp_f = int(flength_samples / 2)
        extend = int(flength_samples / 2) - 1
    else:
        sp_b = int((flength_samples - 1) / 2)
        sp_f = int((flength_samples - 1) / 2)
        extend = int((flength_samples - 1) / 2)

    sig_padded = np.pad(signal, extend, 'reflect')
    # sig_padded = signal
    win = window(flength_samples)
    idx = sp_b

    while (idx + sp_f) < len(sig_padded):
        frame = sig_padded[idx - sp_b:idx + sp_f + 1]
        yield frame * win
        idx += frate_samples


def spliceFeats(feats, context):
    context = int(context)
    frame_num = feats.shape[0]
    feat_dim = feats.shape[1]

    spliced_feats = np.zeros((frame_num, int(feat_dim * (2 * context + 1))))

    feats = np.append(np.zeros((context, feat_dim)), feats, axis=0)
    feats = np.append(feats, np.zeros((context, feat_dim)), axis=0)

    for i in range(0, frame_num - context):
        spliced_feats[i, :] = feats[i:i + 2 * context + 1].reshape(-1)
    return spliced_feats


def createFbank(nfilters, nfft, srate, warp_fact=1):
    mel_max = 2595 * np.log10(1 + (srate / warp_fact) / 1400)
    fwarped = np.linspace(0, mel_max, nfilters + 2)

    mel_filts = np.zeros((nfilters, int(np.floor(nfft / 2 + 1))))
    hz_points = warp_fact * (700 * (10 ** (fwarped / 2595) - 1))
    bin = np.floor((nfft + 1) * hz_points / srate)

    for m in range(1, nfilters + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            mel_filts[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            mel_filts[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return mel_filts

def createHearingFbank(nfft, srate):

    nfilters=20
    hz_points = np.asarray([250, 375, 505, 654, 795, 995, 1130, 1315, 1515, 1720, 1930, 2140, 2355, 2600, 2900, 3255, 3680, 4200, 4860, 5720, 7000])
    bin = np.floor((nfft + 1) * hz_points / srate)
    mel_filts = np.zeros((nfilters, int(np.floor(nfft / 2 + 1))))

    for m in range(1, nfilters + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m_plus = int(bin[m])  # center

        for k in range(f_m_minus, f_m_plus):
            mel_filts[m - 1, k] = 1

    return mel_filts

def createLinearFbank(nfilters, nfft, srate):

    mel_filts = np.zeros((nfilters, int(np.floor(nfft / 2 + 1))))
    hz_points = np.linspace(0, srate/2, nfilters + 2)
    bin = np.floor((nfft + 1) * hz_points / srate)

    for m in range(1, nfilters + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            mel_filts[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            mel_filts[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return mel_filts


def warp_func_bark(x, warp_fact=1):
    return 6 * np.arcsinh((x / warp_fact) / 600)


def createFbankCochlear(nfilters, nfft, srate, om_w=0.2, alp=2.5, fixed=1, bet=2.5, warp_fact=1):
    f_max = srate / 2
    warped_max = warp_func_bark(f_max, warp_fact)
    fwarped_cf = np.linspace(0, warped_max, nfilters)
    f_linear = np.linspace(0, f_max, int(np.floor(nfft / 2 + 1)))
    f_warped = warp_func_bark(f_linear, warp_fact)
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


def computeLpcFast(signal, order, keepreal=True):
    y = np.fft.ifft(np.fft.fft(signal, len(signal)) * np.conj(np.fft.fft(signal, len(signal))))
    if keepreal:
        y = np.real(y)
    xlpc = lpc_solve.solve_toeplitz(y[0:order], -y[1:order + 1])
    xlpc = np.append(1, xlpc)
    gg = y[0] + np.sum(xlpc * y[1:order + 2])

    return xlpc, gg


def computeModSpecFromLpc(gg, xlpc, lim):
    xlpc[1:] = -xlpc[1:]
    lpc_cep = np.zeros(lim, dtype=xlpc.dtype)
    lpc_cep[0] = np.log(np.sqrt(gg))
    lpc_cep[1] = xlpc[1]
    if xlpc.shape[0] < lim:
        xlpc = np.append(xlpc, np.zeros(int(lim - xlpc.shape[0] + 1)))
    for n in range(2, lim):
        aa = np.arange(1, n) / n
        bb = np.flipud(xlpc[1:n])
        cc = lpc_cep[1:n]
        acc = np.sum(aa * bb * cc)
        lpc_cep[n] = acc + xlpc[n]
    return lpc_cep