### FDLP-spectrogram 

This repo contains an implementation of FDLP-spectrogram from the paper 

**Radically Old Way of Computing Spectra: Applications in End-to-End ASR**

https://arxiv.org/abs/2103.14129

The implementation allows fast batch computation of FDLP-spectrogram that can even be used on the fly for DNN training.

To compute FDLP spectrogram

```python

fdlp=fdlp()
# speech (batch x signal length) : padded speech signals formed into a batch
# lens (batch) : lengths of each padded speech siganl in the batch
feats, olens=fdlp.extract_feats(speech,lens)

```

The fdlp class takes the following important parameters which are set to reasonable default values.

```python
 n_filters: int = 80,
 coeff_num: int = 100,
 coeff_range: str = '1,100',
 order: int = 150,
 fduration: float = 1.5,
 frate: int = 100,
 overlap_fraction: float = 0.25,
 srate: int = 16000
```

The performance of an e2e ASR with these features can be found in https://arxiv.org/abs/2103.14129 and is summarized below 

| Data set                                                        |  mel-spectrogram  |  FDLP-spectrogram |
|-----------------------------------------------------------------|:-----------------:|:-----------------:|
| WSJ (test_eval92)                                               |        5.1        |        4.8        |
| REVERB (et_real_1ch / et_real_1ch_wpe / et_real_8ch_beamformit) | 23.2 / 20.7 / 9.2 | 19.4 / 18.0 / 7.2 |
| CHIME4 (et05_real_isolated_1ch_track / et05_real_beamformit_2mics / et05_real_beamformit_5mics) | 23.7 / 20.4 / 16.8 | 23.4 / 19.5 / 15.8 |