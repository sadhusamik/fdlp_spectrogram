import click


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
@click.argument('rspecifier')
@click.argument('wspecifier')
def make_fdlp(n_filters, coeff_num, coeff_range, order, fduration, frate, overlap_fraction, srate):
    pass
