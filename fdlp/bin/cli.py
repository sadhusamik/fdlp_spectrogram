import click
from fdlp import FDLP

@click.group()
def cli():
    pass


@cli.command()
@click.option('--n-filters', default=1,
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
    import librosa
    import kaldi_io

    click.echo(f"make-kaldi-fdlp {wav_scp} {wspecifier}")
    fea_extractor = FDLP(n_filters, coeff_num, coeff_range, order,
        fduration, frate, overlap_fraction, srate=srate)
    click.echo(vars(fea_extractor))

    feat_lens = {}
    with open(wav_scp, "r") as fd_in,\
         kaldi_io.open_or_fd(wspecifier, "wb") as fd_out:
        for line in fd_in:
            utt, audio_f = line.split()
            x, sr = librosa.load(audio_f, sr=srate)
            fea, fea_len = fea_extractor.extract_feats(x, lens=None)
            kaldi_io.write_mat(fd_out, fea, key=utt)
            feat_lens[utt] = fea_len
    
    if utt2num_frames is not None:
        with open(utt2num_frames, "w") as fd:
            for utt, fea_len in fea_lens.items():
                fd.write(f"{utt} {fea_len}\n")

if __name__ == '__main__':
    cli()
