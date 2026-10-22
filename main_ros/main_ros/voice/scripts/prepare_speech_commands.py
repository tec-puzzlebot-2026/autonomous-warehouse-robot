"""
prepare_speech_commands.py - pull training data from Google Speech Commands v0.02.

1. Download (about 2.3 GB) and extract:
     http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
2. Run:
     python prepare_speech_commands.py --sc_dir path/to/speech_commands_v0.02

It copies:
  - your command words (those that exist in the dataset) -> voice/data/google/<word>/
  - a mix of all the OTHER words                          -> voice/data/google/_unknown_/
  - 1-second chunks of the background noise recordings    -> voice/data/google/_unknown_/
"""
import argparse
import glob
import os
import random
import shutil

import numpy as np
import soundfile as sf

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc_dir", required=True)
    ap.add_argument("--max_per_word", type=int, default=1500)
    ap.add_argument("--unknown_words_total", type=int, default=2500)
    ap.add_argument("--noise_chunks", type=int, default=400)
    args = ap.parse_args()
    rng = random.Random(0)

    # --- command words ------------------------------------------------------
    for w in C.WORDS:
        src = os.path.join(args.sc_dir, w)
        if not os.path.isdir(src):
            print(f"[!] '{w}' is not in Speech Commands -> record it with record.py "
                  f"(ask many different people)")
            continue
        files = sorted(glob.glob(os.path.join(src, "*.wav")))
        rng.shuffle(files)
        dst = os.path.join(C.DATA_DIR, C.GOOGLE_SUBDIR, w)
        os.makedirs(dst, exist_ok=True)
        for f in files[:args.max_per_word]:
            shutil.copy(f, os.path.join(dst, os.path.basename(f)))
        print(f"{w:10s}: {min(len(files), args.max_per_word)} files")

    # --- other words -> unknown ---------------------------------------------
    dst = os.path.join(C.DATA_DIR, C.GOOGLE_SUBDIR, C.UNKNOWN)
    os.makedirs(dst, exist_ok=True)
    others = [d for d in sorted(os.listdir(args.sc_dir))
              if os.path.isdir(os.path.join(args.sc_dir, d))
              and not d.startswith("_") and d not in C.WORDS]
    per_word = max(1, args.unknown_words_total // max(1, len(others)))
    n_unk = 0
    for d in others:
        files = sorted(glob.glob(os.path.join(args.sc_dir, d, "*.wav")))
        rng.shuffle(files)
        for f in files[:per_word]:
            spk, _, idx = os.path.basename(f)[:-4].split("_")
            shutil.copy(f, os.path.join(dst, f"{spk}_nohash_{d}{idx}.wav"))
            n_unk += 1
    print(f"unknown words: {n_unk} files from {len(others)} other words")

    # --- background noise -> unknown -----------------------------------------
    noise_files = sorted(glob.glob(os.path.join(args.sc_dir, "_background_noise_", "*.wav")))
    chunks = []
    for i, f in enumerate(noise_files):
        y, sr = sf.read(f, dtype="float32")
        if sr != C.SR:
            continue
        for start in range(0, len(y) - C.SR, C.SR):
            chunks.append((i, start, y[start:start + C.SR]))
    rng.shuffle(chunks)
    for i, start, chunk in chunks[:args.noise_chunks]:
        gain = rng.uniform(0.1, 1.0)
        sf.write(os.path.join(dst, f"noise{i}_nohash_{start}.wav"), chunk * gain, C.SR)
    print(f"noise chunks : {min(len(chunks), args.noise_chunks)}")


if __name__ == "__main__":
    main()
