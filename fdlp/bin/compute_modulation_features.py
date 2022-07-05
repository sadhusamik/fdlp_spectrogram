#!/usr/bin/env python3
# encoding: utf-8

"""
Author: samiksadhu, Johns Hopkins University
"""

import numpy as np
import argparse
import sys
from fdlp.fdlp import FDLP
from fdlp.utils import get_kaldi_ark
from kaldiio import ReadHelper


def get_args():
    parser = argparse.ArgumentParser('Extract Modulation Features (FDLP-spectrogram OR M-vectors)')
    parser.add_argument('scp', help='scp file')
    parser.add_argument('outfile', help='output file')
    parser.add_argument("--segment_file", default=None, type=str, help="segment file will be used if provided")
    parser.add_argument('--n_filters', type=int, default=23, help='number of filters (30)')
    parser.add_argument('--fduration', type=float, default=0.02, help='Window length (0.02 sec)')
    parser.add_argument('--frate', type=int, default=100, help='Frame rate (100 Hz)')
    parser.add_argument('--coeff_num', type=int, default=100, help='Number of modulation coefficients to keep (100)')
    parser.add_argument('--coeff_range', type=str, default='1,100', help='Range of Modulation coefficients to use')
    parser.add_argument('--order', type=int, default=100, help='FDLP model order')
    parser.add_argument('--overlap_fraction', type=float, default=0.15, help='Overlap fraction for overlap-add')
    parser.add_argument('--lifter_file', type=str, default=None,
                        help='Provide lifter file if not using rectangular lifter')
    parser.add_argument('--lfr', type=int, default=10,
                        help='Features are computed at this rate it return_mvector=True then interpolated to frate')
    parser.add_argument('--return_mvector', type=bool, default=False,
                        help='Set to return M-vector instead of FDLP-spectograms')
    parser.add_argument('--complex_mvectors', type=bool, default=False,
                        help='Use complex LPC to compute modulations')
    parser.add_argument('--no_window', type=bool, default=False,
                        help='Set to use rectangular windows over time')
    parser.add_argument('--srate', type=int, default=16000, help='Sampling rate of the signal')
    parser.add_argument('--fbank_type', type=str, default='mel,1',
                        help='mel,warp_fact OR cochlear,om_w,alpa,fixed,beta,warp_fact, OR uniform OR hearing')
    parser.add_argument('--derivative', action='store_true', help='Set to compute derivative of the signal')
    parser.add_argument('--normalize_uttwise_variance', type=bool, default=False,
                        help='Set to perform utterancewise variance normalization')
    parser.add_argument("--write_utt2num_frames", action="store_true", help="Set to write utt2num_frames")

    return parser.parse_args()


def compute_modulations(args):
    # Define FDLP class
    feat_model = FDLP(n_filters=args.n_filters, coeff_num=args.coeff_num, coeff_range=args.coeff_range,
                      order=args.order, normalize_uttwise_variance=args.normalize_uttwise_variance,
                      fduration=args.fduration, frate=args.frate, overlap_fraction=args.overlap_fraction,
                      lifter_file=args.lifter_file, lfr=args.lfr, return_mvector=args.return_mvector,
                      complex_mvectors=args.complex_mvectors, no_window=args.no_window, srate=args.srate)

    if args.write_utt2num_frames:
        all_lens = {}
    all_feats = {}

    if args.segment_file is None:
        with ReadHelper('scp:' + args.scp) as reader:
            for key, (rate, signal) in reader:

                signal = signal / np.power(2, 15)
                print('%s: Computing Features for file: %s' % (sys.argv[0], key))
                sys.stdout.flush()

                feats, _ = feat_model.extract_feats(signal[np.newaxis, :])
                feats = feats[0]
                all_feats[key] = feats
                if args.write_utt2num_frames:
                    all_lens[key] = feats.shape[0]
    else:
        with ReadHelper('scp:' + args.scp, segments=args.segment_file) as reader:
            for key, (rate, signal) in reader:

                print('%s: Computing Features for file: %s' % (sys.argv[0], key))
                sys.stdout.flush()

                feats, _ = feat_model.extract_feats(signal[np.newaxis, :])
                feats = feats[0]
                all_feats[key] = feats
                if args.write_utt2num_frames:
                    all_lens[key] = feats.shape[0]

        get_kaldi_ark(all_feats, args.outfile)

        if args.write_utt2num_frames:
            with open(args.outfile + '.len', 'w+') as file:
                for key, lens in all_lens.items():
                    p = "{:s} {:d}".format(key, lens)
                    file.write(p)
                    file.write("\n")


if __name__ == '__main__':
    args = get_args()
    compute_modulations(args)
