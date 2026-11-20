"""
augment.py - make many varied copies of the TEAM recordings, so the HMMs see
"new voices, new rooms, new microphones" without recording more.

Run from the voice/ folder (same place as train.py):

    python3 -u scripts/augment.py --exclude elias        # for the honest test (Elias stays unseen)
    python3 -u scripts/augment.py --clean                # for the final model (everyone)

Options:
    --copies N         augmented copies per command-word clip      (default 24)
    --unknown-copies N augmented copies per team "_unknown_" clip  (default 8)
    --exclude A B ..   speakers NOT to augment (your test speakers!)
    --noise-dir DIR    extra folder(s) with .wav noise (e.g. a recording of the
                       robot driving, a noisy lab). Can be given several times.
    --clean            delete data/augmented/ first and rebuild it
    --seed S           random seed (same seed -> same copies)

What one copy gets (each effect is applied with some probability, random strength):
    speed      0.85-1.15x  (tempo AND pitch, like a faster/slower or higher/lower speaker)
    pitch      +-2.5 semitones  (different voice height, same timing)
    tempo      0.85-1.20x  (talking faster/slower, same voice)
    mic EQ     random high-pass / low-pass  (cheap phone mic, laptop mic, far mic)
    reverb     synthetic room, 0.15-0.6 s  (echoey warehouse vs small room)
    noise      real noise (Google background clips, your --noise-dir) or white/pink/
               brown noise, at 10-30 dB SNR (--snr-min / --snr-max)
    gain       -12..+4 dB  (quiet and loud speakers), never clipped
    padding    0-0.1 s of silence before and after

Output: data/augmented/<word>/<speaker>_aug<k>_<original name>.wav
The file name still starts with the ORIGINAL speaker, so train.py's speaker-based
train/test split keeps working. Google clips are never augmented (they are
already thousands of speakers).

IMPORTANT: in config.py raise MAX_TRAIN_PER_CLASS (e.g. 5000). Otherwise the
augmented team clips fill the 1000-clip cap and the Google clips (many voices
for "stop" and _unknown_) get pushed out.
"""
import argparse
import glob
import os
import shutil
import sys
import time
import warnings
import zlib
from multiprocessing import Pool

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# harmless numba message from librosa on leading/trailing silence
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

AUG_DIR = os.path.join(C.DATA_DIR, "augmented")
GOOGLE_DIR = os.path.join(C.DATA_DIR, getattr(C, "GOOGLE_SUBDIR", "google"))
NOISE = []          # filled per worker: list of float32 arrays at C.SR


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def speaker_of(path):
    return os.path.basename(path).split("_")[0].lower()


def inside(path, folder):
    return os.path.abspath(path).startswith(os.path.abspath(folder) + os.sep)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def colored_noise(n, kind, rng):
    w = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return w
    spec = np.fft.rfft(w)
    f = np.arange(len(spec)) + 1.0
    spec /= np.sqrt(f) if kind == "pink" else f          # pink 1/f, brown 1/f^2 power
    out = np.fft.irfft(spec, n).astype(np.float32)
    return out / (np.abs(out).max() + 1e-9)


def get_noise(n, rng):
    """n samples of noise: real recordings when available, else colored noise."""
    if NOISE and rng.random() < 0.7:
        src = NOISE[rng.integers(len(NOISE))]
        if len(src) < n:
            src = np.tile(src, int(np.ceil(n / len(src))) + 1)
        start = rng.integers(0, len(src) - n + 1)
        return src[start:start + n].copy()
    return colored_noise(n, rng.choice(["white", "pink", "brown"]), rng)


# ----------------------------------------------------------------------------
# effects
# ----------------------------------------------------------------------------
def speed(y, rng):
    f = rng.uniform(0.85, 1.15)                           # resample = tempo + pitch
    return librosa.resample(y, orig_sr=C.SR, target_sr=int(C.SR / f))


def pitch(y, rng):
    return librosa.effects.pitch_shift(y, sr=C.SR, n_steps=rng.uniform(-2.5, 2.5))


def tempo(y, rng):
    return librosa.effects.time_stretch(y, rate=rng.uniform(0.85, 1.2))


def mic_eq(y, rng):
    if rng.random() < 0.7:
        sos = butter(2, rng.uniform(60, 300), "highpass", fs=C.SR, output="sos")
        y = sosfilt(sos, y)
    if rng.random() < 0.7:
        sos = butter(4, rng.uniform(3000, 7000), "lowpass", fs=C.SR, output="sos")
        y = sosfilt(sos, y)
    return y.astype(np.float32)


def reverb(y, rng):
    rt60 = rng.uniform(0.15, 0.6)
    n = int(rt60 * C.SR)
    t = np.arange(n) / C.SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    ir[0] = 1.0
    ir /= np.abs(ir).sum() ** 0.5
    wet = fftconvolve(y, ir)[:len(y) + n // 2]
    dry = np.pad(y, (0, len(wet) - len(y)))
    mix = rng.uniform(0.2, 0.6)
    out = (1 - mix) * dry + mix * wet * (rms(y) / (rms(wet) + 1e-9))
    return out.astype(np.float32)


SNR_RANGE = (10.0, 30.0)


def add_noise(y, rng):
    speech_rms = rms(librosa.effects.trim(y, top_db=C.TRIM_DB)[0])
    snr = rng.uniform(*SNR_RANGE)
    n = get_noise(len(y), rng)
    n *= speech_rms / (rms(n) * 10 ** (snr / 20) + 1e-12)
    return (y + n).astype(np.float32)


def pad(y, rng):
    a, b = (int(rng.uniform(0, 0.1) * C.SR) for _ in range(2))
    return np.pad(y, (a, b))


def augment_once(y, rng):
    """One random combination of effects. At least one 'voice' effect is used."""
    voice_fx = [(speed, 0.6), (pitch, 0.5), (tempo, 0.3)]
    used = False
    for fx, p in voice_fx:
        if rng.random() < p:
            y = fx(y, rng)
            used = True
    if not used:
        y = speed(y, rng)
    if rng.random() < 0.4:
        y = mic_eq(y, rng)
    if rng.random() < 0.4:
        y = reverb(y, rng)
    y = pad(y, rng)
    if rng.random() < 0.5:
        y = add_noise(y, rng)
    y = y * 10 ** (rng.uniform(-12, 4) / 20)              # gain
    peak = np.abs(y).max()
    if peak > 0.99:                                       # never clip
        y = y * (0.99 / peak)
    return y.astype(np.float32)


# ----------------------------------------------------------------------------
# workers
# ----------------------------------------------------------------------------
def init_worker(noise_files, snr_range):
    global NOISE, SNR_RANGE
    SNR_RANGE = snr_range
    NOISE = []
    for f in noise_files:
        try:
            y, _ = librosa.load(f, sr=C.SR, mono=True)
            if len(y) > C.SR // 4 and rms(y) > 1e-4:
                NOISE.append(y.astype(np.float32))
        except Exception:
            pass


def job(task):
    """task = (source path, label, number of copies, seed). Returns #written."""
    src, label, copies, seed = task
    try:
        y, _ = librosa.load(src, sr=C.SR, mono=True)
    except Exception:
        return 0
    if len(y) < C.MIN_SAMPLES:
        return 0
    out_dir = os.path.join(AUG_DIR, label)
    stem = os.path.splitext(os.path.basename(src))[0]
    spk = speaker_of(src)
    written = 0
    for k in range(copies):
        dst = os.path.join(out_dir, f"{spk}_aug{k}_{stem}.wav")
        if os.path.exists(dst):                           # resume / re-run safe
            continue
        rng = np.random.default_rng([seed, zlib.crc32(src.encode()), k])
        sf.write(dst, augment_once(y.copy(), rng), C.SR, subtype="PCM_16")
        written += 1
    return written


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=24)
    ap.add_argument("--unknown-copies", type=int, default=8)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--noise-dir", action="append", default=[])
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr-min", type=float, default=10.0, help="strongest noise (dB SNR)")
    ap.add_argument("--snr-max", type=float, default=30.0, help="weakest noise (dB SNR)")
    args = ap.parse_args()
    exclude = {s.lower() for s in args.exclude}

    if args.clean and os.path.isdir(AUG_DIR):
        shutil.rmtree(AUG_DIR)
        print(f"Removed old {AUG_DIR}")
    labels = C.WORDS + [C.UNKNOWN]
    for lab in labels:
        os.makedirs(os.path.join(AUG_DIR, lab), exist_ok=True)

    # previous augmented copies of now-excluded speakers would leak into the test
    for lab in labels:
        for f in glob.glob(os.path.join(AUG_DIR, lab, "*.wav")):
            if speaker_of(f) in exclude:
                os.remove(f)

    # ---- source clips: team recordings only ------------------------------------
    tasks, per_label, skipped = [], {lab: 0 for lab in labels}, 0
    for lab in labels:
        for f in sorted(glob.glob(os.path.join(C.DATA_DIR, "**", lab, "*.wav"), recursive=True)):
            if inside(f, AUG_DIR) or inside(f, GOOGLE_DIR):
                continue
            if speaker_of(f) in exclude:
                skipped += 1
                continue
            n = args.unknown_copies if lab == C.UNKNOWN else args.copies
            tasks.append((f, lab, n, args.seed))
            per_label[lab] += 1

    # ---- noise sources ------------------------------------------------------------
    noise_files = sorted(set(
        glob.glob(os.path.join(C.DATA_DIR, "**", "noise*_nohash_*.wav"), recursive=True)
        + glob.glob(os.path.join(C.DATA_DIR, "**", "_background_noise_", "*.wav"), recursive=True)
        + [f for d in args.noise_dir for f in glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)]
    ))

    print(f"Team clips to augment: {len(tasks)}  (skipped {skipped} from excluded speakers {sorted(exclude) or '-'})")
    for lab in labels:
        n = args.unknown_copies if lab == C.UNKNOWN else args.copies
        print(f"  {lab:11s} {per_label[lab]:4d} clips x {n:2d} copies = {per_label[lab] * n:6d}")
    print(f"Noise recordings available: {len(noise_files)} (+ synthetic white/pink/brown noise)")
    if not tasks:
        raise SystemExit("No team recordings found - run from the voice/ folder.")

    t0 = time.time()
    workers = max(1, (os.cpu_count() or 2) - 1)
    total = 0
    with Pool(workers, initializer=init_worker, initargs=(noise_files, (args.snr_min, args.snr_max))) as pool:
        for w in tqdm(pool.imap_unordered(job, tasks, chunksize=4), total=len(tasks),
                      desc="  augmenting", unit="clip", dynamic_ncols=True):
            total += w
    have = sum(len(glob.glob(os.path.join(AUG_DIR, lab, "*.wav"))) for lab in labels)
    print(f"\nWrote {total} new files in {time.time() - t0:.0f} s  ->  {have} augmented files in {AUG_DIR}")
    cap = getattr(C, "MAX_TRAIN_PER_CLASS", 1000)
    biggest = max(len(glob.glob(os.path.join(AUG_DIR, lab, "*.wav"))) for lab in labels)
    if cap < biggest + 1000:
        print("\n" + "!" * 70)
        print(f"[!] config.py MAX_TRAIN_PER_CLASS = {cap} is TOO LOW. Set it to at least "
              f"{biggest + 1500} BEFORE training,\n    or the Google clips (thousands of voices "
              f"for 'stop' and _unknown_) are dropped and\n    the rejection threshold breaks.")
        print("!" * 70)


if __name__ == "__main__":
    main()