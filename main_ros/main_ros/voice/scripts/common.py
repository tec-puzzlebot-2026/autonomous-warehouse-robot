"""
common.py - feature extraction and scoring, shared by train.py and recognize.py.

Pipeline for one utterance:
  raw audio -> trim silence -> pre-emphasis -> MFCC (13) + delta (13)
  + delta-delta (13) = 39 numbers per 10 ms frame -> per-utterance
  normalisation (CMVN).
The HMMs never see raw audio, only this (T x 39) matrix.
"""
import numpy as np
import librosa

try:                       # imported inside the ROS 2 package
    from . import config as C
except ImportError:        # run as a plain script from this folder
    import config as C


def load_wav(path):
    y, _ = librosa.load(path, sr=C.SR, mono=True)
    return y


def trim_silence(y):
    yt, _ = librosa.effects.trim(y, top_db=C.TRIM_DB,
                                 frame_length=C.WIN, hop_length=C.HOP)
    return yt if len(yt) >= C.MIN_SAMPLES else y


def extract_features(y, trim=True):
    """Return a (T, 39) float64 feature matrix for one utterance."""
    y = np.asarray(y, dtype=np.float32)
    if trim:
        y = trim_silence(y)
    if len(y) < C.MIN_SAMPLES:
        y = np.pad(y, (0, C.MIN_SAMPLES - len(y)))

    # Pre-emphasis boosts high frequencies (consonants like s, t, f).
    y = librosa.effects.preemphasis(y)

    mfcc = librosa.feature.mfcc(y=y, sr=C.SR, n_mfcc=C.N_MFCC,
                                n_fft=C.WIN, hop_length=C.HOP,
                                n_mels=C.N_MELS)
    d1 = librosa.feature.delta(mfcc, order=1, mode="nearest")   # velocity
    d2 = librosa.feature.delta(mfcc, order=2, mode="nearest")   # acceleration
    feats = np.vstack([mfcc, d1, d2]).T                         # (T, 39)

    # Cepstral Mean & Variance Normalisation: removes the "colour" of the
    # microphone/room and loudness differences -> helps speaker/mic independence.
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
    return feats.astype(np.float64)


def score_utterance(bundle, X):
    """
    Forward algorithm on every model.
    Returns per-frame log-likelihoods for each word, and for the garbage model.
    Dividing by the number of frames makes long and short words comparable.
    """
    n = len(X)
    word_ll = np.array([m.score(X) / n for m in bundle["models"]])
    garbage_ll = bundle["garbage"].score(X) / n
    return word_ll, garbage_ll


def decide(word_ll, garbage_ll, threshold):
    """
    Pick the best word, but only accept it if it beats the garbage model
    by more than `threshold` (log-likelihood per frame).
    Returns (index or None, margin).
    """
    best = int(np.argmax(word_ll))
    margin = float(word_ll[best] - garbage_ll)
    return (best if margin > threshold else None), margin
