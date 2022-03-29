import click
import sys
from fdlp import FDLP

@click.group()
def cli():
    pass


@cli.command()
@click.option('--n-filters', default=80,
	help='Number of filters'
)
@click.option('--coeff-num', default=100,
	help='Number of modulation coefficients to compute'
)
@click.option('--coeff-range', default='1,100',
	help='Range of modulation coefficients to preserve'
)
@click.option('--order', default=150,
	help='Order of FDLP model'
)
@click.option('--fduration', default=1.5,
	help='Duration of window in seconds'
)
@click.option('--frate', default=100,
	help='Frame rate'
)
@click.option('--overlap-fraction', default=0.25,
	help='Overlap fraction in Overlap-Add'
)
@click.option('--srate', default=16000,
	help='Sample rate of the speech signal'
)
@click.argument('wav_scp')
@click.argument('wspecifier')
@click.argument('utt2num_frames', required=False)
def kaldi(wav_scp, wspecifier, n_filters, coeff_num, coeff_range, order, fduration, frate, overlap_fraction, srate, utt2num_frames=None):
    """
    Make FDLP features from wav.scp with utt-level recordings.

    The features are stored in kaldi-like wspecifier,e.g.
    \"ark:| copy-feats --compress=true ark:- ark,scp:data/feats.ark,data/feats.scp\"

    You can also specify optional `utt2num_frams` file to store feature lenghts
    for each utterance.
    """
    import soundfile as sf
    import kaldi_io
    import numpy as np

    click.echo(f"make-fdlp kaldi {wav_scp} {wspecifier}")
    fea_extractor = FDLP(n_filters, coeff_num, coeff_range, order,
        fduration, frate, overlap_fraction, srate=srate)
    click.echo(vars(fea_extractor))

    feat_lens = {}
    err_cnt = 0
    cnt = 0
    with open(wav_scp, "r") as fd_in,\
         kaldi_io.open_or_fd(wspecifier, "wb") as fd_out:
        for line in fd_in:
            cnt += 1
            utt, audio_f = line.split()
            offset = 0
            if len(audio_f.split(":")) == 2:
                audio_f, offset = audio_f.split(":")
                offset = int(offset)

            with open(audio_f, "rb") as f:
                f.seek(offset)
                x, sr = sf.read(f)

            x = np.expand_dims(x, axis=0)
            try:
                fea, fea_len = fea_extractor.extract_feats(x, ilens=None)
            except Exception as e:
                err_cnt += 1
                print("WARNING: Failed to extract feats from utt \"" + utt + "\" :" + str(e), file=sys.stderr)
                continue

            kaldi_io.write_mat(fd_out, np.squeeze(fea, axis=0), key=utt)
            feat_lens[utt] = fea_len
    
    if utt2num_frames is not None:
        with open(utt2num_frames, "w") as fd:
            for utt, fea_len in feat_lens.items():
                fd.write(f"{utt} {fea_len}\n")

    print(f"Failed to extract FDLP features in {err_cnt}/{cnt} cases.", file=sys.stderr)

if __name__ == '__main__':
    cli()
