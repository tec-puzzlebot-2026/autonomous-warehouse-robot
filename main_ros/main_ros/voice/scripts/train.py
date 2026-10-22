"""
train.py - train one HMM per command word + one "garbage" HMM, test them,
choose the rejection threshold, and save everything to models.pkl.

Usage:
    python train.py
"""
import glob
import os
import pickle
import random
import time
import zlib
from multiprocessing import Pool

import numpy as np
from hmmlearn import hmm
from hmmlearn.base import ConvergenceMonitor
from sklearn.cluster import KMeans
from tqdm import tqdm

import config as C
from common import load_wav, extract_features, score_utterance, decide
# (train.py is always run as a script: python3 train.py)


# ----------------------------------------------------------------------------
# 1. Data loading with a SPEAKER-based train/test split
# ----------------------------------------------------------------------------
def speaker_of(path):
    return os.path.basename(path).split("_")[0]


def is_google(path):
    google_dir = os.path.join(C.DATA_DIR, C.GOOGLE_SUBDIR) + os.sep
    return os.path.abspath(path).startswith(os.path.abspath(google_dir))


def is_test(path):
    """Same speaker always lands in the same split, so test results measure
    performance on voices the model never heard.
    Team clips: test = the speakers listed in TEST_SPEAKERS.
    Google clips (thousands of speakers): automatic hash split."""
    spk = speaker_of(path)
    if C.TEST_SPEAKERS and not is_google(path):
        return spk in [s.lower() for s in C.TEST_SPEAKERS]
    return zlib.crc32(spk.encode()) % 100 < C.TEST_FRACTION * 100


def split_class(name):
    """Return (train_files, test_files) for one class (no audio loaded yet)."""
    files = sorted(set(glob.glob(os.path.join(C.DATA_DIR, "**", name, "*.wav"),
                                 recursive=True)))
    if not files:
        raise SystemExit(f"No .wav files in any '{name}' folder under {C.DATA_DIR}")
    test = [f for f in files if is_test(f)]
    train = files if C.USE_ALL_FOR_FINAL else [f for f in files if not is_test(f)]
    rng = random.Random(0)
    rng.shuffle(train)
    rng.shuffle(test)

    # Always keep every team clip; Google clips only fill up to the cap.
    own = [f for f in train if not is_google(f)]
    google = [f for f in train if is_google(f)]
    train = own + google[:max(0, C.MAX_TRAIN_PER_CLASS - len(own))]
    test = test[:C.MAX_TEST_PER_CLASS]

    spk = sorted({speaker_of(f) for f in own})
    print(f"  {name:11s} team clips: {len(own):4d} ({', '.join(spk)})  "
          f"google: {len(train) - len(own):4d}  test: {len(test):4d}")
    return train, test


# ----------------------------------------------------------------------------
# 1b. Feature extraction: parallel (all CPU cores) + cached on disk
# ----------------------------------------------------------------------------
CACHE_PATH = os.path.join(C.VOICE_DIR, "feature_cache.pkl")
FEAT_SIG = (C.SR, C.N_MFCC, C.N_MELS, C.WIN, C.HOP, C.TRIM_DB, C.MIN_SAMPLES)


def _file_key(path):
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


def _extract(path):
    return path, extract_features(load_wav(path)).astype(np.float32)


def extract_all(paths):
    """Features for every path. Already-processed files come from the cache,
    the rest are computed in parallel with a progress bar."""
    cache = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                saved = pickle.load(f)
            if saved.get("sig") == FEAT_SIG:
                cache = saved["feats"]
        except Exception:
            cache = {}

    feats, todo = {}, []
    for p in paths:
        k = _file_key(p)
        if k in cache:
            feats[p] = cache[k]
        else:
            todo.append(p)
    print(f"  {len(paths) - len(todo)} files from cache, {len(todo)} to extract")

    if todo:
        workers = max(1, (os.cpu_count() or 2) - 1)
        with Pool(workers) as pool:
            for p, x in tqdm(pool.imap_unordered(_extract, todo, chunksize=8),
                             total=len(todo), desc="  extracting features",
                             unit="file", dynamic_ncols=True):
                feats[p] = x
                cache[_file_key(p)] = x
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"sig": FEAT_SIG, "feats": cache}, f)
    return {p: feats[p].astype(np.float64) for p in paths}


class BarMonitor(ConvergenceMonitor):
    """hmmlearn calls report() once per Baum-Welch iteration -> update a bar."""
    def __init__(self, tol, n_iter, bar):
        super().__init__(tol, n_iter, verbose=False)
        self.bar = bar

    def report(self, log_prob):
        super().report(log_prob)
        self.bar.update(1)
        self.bar.set_postfix(loglik=f"{log_prob:,.0f}")


# ----------------------------------------------------------------------------
# 2. Model construction
# ----------------------------------------------------------------------------
def left_to_right_transmat(n):
    """State i can only stay (self-loop) or move to i+1. Speech flows forward
    in time, so states never go backwards. Zeros stay zero during training."""
    T = np.zeros((n, n))
    for i in range(n - 1):
        T[i, i], T[i, i + 1] = 0.6, 0.4
    T[-1, -1] = 1.0
    return T


def flat_start_init(model, seqs, n_states, n_mix):
    """'Uniform segmentation' initialisation: cut every training example into
    n_states equal pieces, piece k feeds state k. Then cluster each state's
    frames into n_mix Gaussians with k-means. This gives Baum-Welch a sensible
    starting point (state 0 ~ start of word, last state ~ end of word)."""
    d = seqs[0].shape[1]
    buckets = [[] for _ in range(n_states)]
    for X in seqs:
        bounds = np.linspace(0, len(X), n_states + 1).astype(int)
        for s in range(n_states):
            seg = X[bounds[s]:bounds[s + 1]]
            if len(seg):
                buckets[s].append(seg)

    means = np.zeros((n_states, n_mix, d))
    covars = np.ones((n_states, n_mix, d))
    weights = np.full((n_states, n_mix), 1.0 / n_mix)
    for s in range(n_states):
        F = np.vstack(buckets[s])
        km = KMeans(n_clusters=n_mix, n_init=3, random_state=0).fit(F)
        for k in range(n_mix):
            pts = F[km.labels_ == k]
            means[s, k] = km.cluster_centers_[k]
            covars[s, k] = (pts.var(axis=0) if len(pts) > 1 else F.var(axis=0)) + 1e-2
            weights[s, k] = max(len(pts), 1) / len(F)
        weights[s] /= weights[s].sum()
    model.means_, model.covars_, model.weights_ = means, covars, weights


def _fit_with_bar(model, seqs, label):
    with tqdm(total=C.N_ITER, desc=f"  {label:11s}", unit="iter",
              leave=True, dynamic_ncols=True) as bar:
        model.monitor_ = BarMonitor(model.tol, model.n_iter, bar)
        t = time.time()
        model.fit(np.vstack(seqs), [len(x) for x in seqs])   # Baum-Welch
        bar.set_postfix(iters=model.monitor_.iter,
                        converged=model.monitor_.converged,
                        secs=f"{time.time() - t:.0f}")
    # Detach the progress bar so the model can be saved with pickle.
    model.monitor_ = ConvergenceMonitor(model.tol, model.n_iter, False)
    return model


def train_word_model(seqs, label):
    n, m = C.WORD_STATES, C.WORD_MIX
    model = hmm.GMMHMM(n_components=n, n_mix=m, covariance_type="diag",
                       n_iter=C.N_ITER, tol=1e-3, random_state=0,
                       init_params="",        # we initialise ourselves
                       params="stmcw")        # but EM may update everything
    model.startprob_ = np.eye(n)[0]           # always start in state 0
    model.transmat_ = left_to_right_transmat(n)
    flat_start_init(model, seqs, n, m)
    return _fit_with_bar(model, seqs, label)


def train_garbage_model(seqs, label="garbage"):
    """Ergodic (fully connected) HMM: any state -> any state. It learns a
    generic 'any speech or noise' model. A real command should fit its own
    word model clearly better than this generic one."""
    model = hmm.GMMHMM(n_components=C.GARBAGE_STATES, n_mix=C.GARBAGE_MIX,
                       covariance_type="diag", n_iter=C.N_ITER, tol=1e-3,
                       random_state=0)
    return _fit_with_bar(model, seqs, label)


# ----------------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"Data folder: {C.DATA_DIR}")
    print(f"Test speakers: {C.TEST_SPEAKERS or 'automatic'}   "
          f"USE_ALL_FOR_FINAL={C.USE_ALL_FOR_FINAL}")

    print("\n[1/4] Finding files")
    splits = {name: split_class(name) for name in C.WORDS + [C.UNKNOWN]}

    print("\n[2/4] Extracting features (MFCC)")
    t1 = time.time()
    all_paths = sorted({p for tr, te in splits.values() for p in tr + te})
    feats = extract_all(all_paths)
    data = {name: ([feats[p] for p in tr], [feats[p] for p in te])
            for name, (tr, te) in splits.items()}
    print(f"  done in {time.time() - t1:.0f} s")

    print(f"\n[3/4] Training {len(C.WORDS)} word HMMs + 1 garbage HMM "
          f"(max {C.N_ITER} Baum-Welch iterations each, stops early when converged)")
    t2 = time.time()
    models = []
    for k, w in enumerate(C.WORDS, 1):
        models.append(train_word_model(data[w][0], f"{k:2d}/11 {w}"))
    garbage = train_garbage_model(data[C.UNKNOWN][0], "11/11 garbage")
    bundle = {"words": C.WORDS, "models": models, "garbage": garbage, "threshold": 0.0}
    print(f"  done in {time.time() - t2:.0f} s")

    print("\n[4/4] Evaluating")

    # ---- Evaluate on held-out speakers --------------------------------------
    n = len(C.WORDS)
    confusion = np.zeros((n, n), dtype=int)
    pos_margins, neg_margins = [], []
    for i, w in enumerate(C.WORDS):
        for X in tqdm(data[w][1], desc=f"  scoring {w:10s}", leave=False,
                      dynamic_ncols=True):
            wl, gl = score_utterance(bundle, X)
            best = int(np.argmax(wl))
            confusion[i, best] += 1
            if best == i:
                pos_margins.append(wl[best] - gl)
    for X in tqdm(data[C.UNKNOWN][1], desc="  scoring _unknown_", leave=False,
                  dynamic_ncols=True):
        wl, gl = score_utterance(bundle, X)
        neg_margins.append(wl.max() - gl)

    if confusion.sum() == 0 or not neg_margins:
        print("\n[!] No test speakers found - set TEST_SPEAKERS in config.py "
              "(e.g. one teammate) to get honest accuracy numbers.")
    acc = np.trace(confusion) / max(confusion.sum(), 1)
    print(f"\nClosed-set accuracy (which of the 10 words?): {acc:.1%}")
    print("Confusion matrix (rows = truth, cols = predicted):")
    print("          " + " ".join(f"{w[:5]:>5s}" for w in C.WORDS))
    for i, w in enumerate(C.WORDS):
        print(f"{w[:9]:>9s} " + " ".join(f"{c:5d}" for c in confusion[i]))

    # ---- Pick the rejection threshold ----------------------------------------
    # Let at most TARGET_FALSE_ACCEPT of the non-command clips through.
    if neg_margins:
        threshold = float(np.percentile(neg_margins, 100 * (1 - C.TARGET_FALSE_ACCEPT)))
    else:
        threshold = C.DEFAULT_THRESHOLD
        neg_margins = [threshold]
    bundle["threshold"] = threshold
    kept = np.mean(np.array(pos_margins) > threshold) if pos_margins else 0.0
    print(f"\nRejection threshold (log-lik/frame margin): {threshold:.3f}")
    print(f"  non-commands wrongly accepted : {np.mean(np.array(neg_margins) > threshold):.1%}")
    print(f"  correct commands still accepted: {kept:.1%}")
    print(f"  => end-to-end command detection rate ~ {acc * kept:.1%}")

    with open(C.MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved {C.MODEL_PATH}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()