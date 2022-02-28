# Modulation Features for Automatic Speech Recognition

This repo contains an implementation of

- **FDLP-spectrogram** from the paper 
**Radically Old Way of Computing Spectra: Applications in End-to-End ASR**
(https://arxiv.org/abs/2103.14129)
- **Modulation Vectors** from the paper **M-vectors: Sub-band Based Energy Modulation Features for Multi-stream Automatic Speech Recognition**
(https://ieeexplore.ieee.org/abstract/document/8682710)

## FDLP-spectrogram

The implementation allows fast batch computation of FDLP-spectrogram that can even be used on the fly for DNN training.

To compute FDLP spectrogram

### Python

```python
from fdlp import FDLP
fdlp = FDLP()
# speech (batch x signal length) : padded speech signals formed into a batch
# lens (batch) : lengths of each padded speech siganl in the batch
# set lens=None if you are computing features one utterance at a time and not as a batch
feats, olens = fdlp.extract_feats(speech, lens)

```

The fdlp class takes the following important parameters which are set to reasonable default values.

```python
 n_filters: int = 80, # Number of filters
 coeff_num: int = 100, # Number of modulation coefficients to compute
 coeff_range: str = '1,100', # Range of modulation coefficients to preserve 
 order: int = 150, # Order of FDLP model
 fduration: float = 1.5, # Duration of window in seconds
 frate: int = 100, # Frame rate
 overlap_fraction: float = 0.25,  # Overlap fraction in Overlap-Add
 srate: int = 16000    # Sample rate of the speech signal
```

### CLI

```
# Kaldi-like features
make-fdlp kaldi wav.scp "ark:| copy-feats ark:- ark,scp:/path/to/srotage/make_fdlp.ark,data/feats.scp" [data/utt2num_frames]
```

For more info type:
```
make-fdlp kaldi --help
```

### Results

The performance of an e2e ASR with these features can be found in https://arxiv.org/abs/2103.14129 and is summarized below 

| Data set                                                        |  mel-spectrogram  |  FDLP-spectrogram |
|-----------------------------------------------------------------|:-----------------:|:-----------------:|
| WSJ (test_eval92)                                               |        5.1        |        4.8        |
| REVERB (et_real_1ch / et_real_1ch_wpe / et_real_8ch_beamformit) | 23.2 / 20.7 / 9.2 | 19.4 / 18.0 / 7.2 |
| CHIME4 (et05_real_isolated_1ch_track / et05_real_beamformit_2mics / et05_real_beamformit_5mics) | 23.7 / 20.4 / 16.8 | 23.4 / 19.5 / 15.8 |

## Modulation vector (M-vector)

```python
from fdlp import FDLP
fdlp = FDLP(lfr=10, return_mvector=True)
# speech (batch x signal length) : padded speech signals formed into a batch
# lens (batch) : lengths of each padded speech siganl in the batch
feats, olens = fdlp.extract_feats(speech, lens)

```

The fdlp class takes the following important parameters for M-vector computation.

```python
 n_filters: int = 80, # Number of filters
 coeff_num: int = 100, # Number of modulation coefficients to compute
 coeff_range: str = '1,100', # Range of modulation coefficients to preserve 
 order: int = 150, # Order of FDLP model
 fduration: float = 1.5, # Duration of window in seconds
 frate: int = 100, # Frame rate
 lfr: int = 10, # M-vectors are computed at this frame-rate and then interpolated to frate
 overlap_fraction: float = 0.25,  # Overlap fraction in Overlap-Add
 srate: int = 16000    # Sample rate of the speech signal
```
Results with these features for Kaldi TDNN models for REVERB data set can be found in **Modulation Vectors as Robust Feature Representation for ASR in Domain Mismatched Conditions** (https://www.isca-speech.org/archive_v0/Interspeech_2019/pdfs/2723.pdf)

## Installation

### Pip

To install the latest, unreleased version, do:

```
pip install git+https://github.com/sadhusamik/fdlp_spectrogram
```

