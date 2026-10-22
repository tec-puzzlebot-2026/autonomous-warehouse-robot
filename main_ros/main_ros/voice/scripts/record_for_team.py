"""
record_for_team.py - standalone data recorder (for teammates AND for you).

Setup (once):
    pip install sounddevice soundfile numpy

Run:
    python record_for_team.py --name yourname

For each clip:  press Enter -> say the word -> press Enter again.
It asks for each command word 10 times plus 30 "not a command" clips,
in random order. Silence before/after the word is fine (it is trimmed
automatically during training).

At the end it creates  recordings_<yourname>.zip  -> send that file.
If you stop halfway (Ctrl+C), run the same command again: it continues
where you left off.
"""
import argparse
import os
import random
import re
import shutil
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

# !!! Must be EXACTLY the same list as WORDS in config.py !!!
WORDS = ["start", "stop", "pause", "resume", "mapping",
         "restocking", "faster", "slower", "return", "status"]
UNKNOWN = "_unknown_"

SR = 16000
MAX_SECONDS = 5.0      # safety limit if someone forgets to press Enter
TAIL_SECONDS = 0.25    # keep recording a bit after Enter so the word end isn't cut
MIN_SECONDS = 0.3
OUT_DIR = "data"

# Words that are NOT commands (and don't sound too much like them).
FILLER_WORDS = ["hello", "table", "banana", "robot", "window", "seven",
                "water", "happy", "paper", "orange", "computer", "box",
                "shelf", "maybe", "thanks", "music", "coffee", "green",
                "monday", "hola", "mesa", "gracias", "bueno", "kitchen",
                "pencil", "yellow", "tomorrow", "listen"]


def count_existing(label, name):
    folder = os.path.join(OUT_DIR, label)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.startswith(name + "_nohash_")])


def save(label, name, audio):
    folder = os.path.join(OUT_DIR, label)
    os.makedirs(folder, exist_ok=True)
    n = 0
    while os.path.exists(os.path.join(folder, f"{name}_nohash_{n}.wav")):
        n += 1
    path = os.path.join(folder, f"{name}_nohash_{n}.wav")
    sf.write(path, audio, SR, subtype="PCM_16")
    return path


def record_until_enter():
    """Record from the mic until the user presses Enter (or MAX_SECONDS)."""
    chunks = []

    def callback(indata, frames, time_info, status):
        if sum(len(c) for c in chunks) < MAX_SECONDS * SR:
            chunks.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=callback):
        input("   RECORDING... say the word, then press Enter ")
        time.sleep(TAIL_SECONDS)
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="your name (letters/numbers only)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--unknown", type=int, default=30)
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    if args.device is not None:
        sd.default.device = (args.device, None)

    name = re.sub(r"[^A-Za-z0-9]", "", args.name).lower()
    if not name:
        raise SystemExit("Please give a name with letters/numbers.")

    # ---- build the list of clips still missing (lets you resume) -----------
    trials = []
    for w in WORDS:
        trials += [(w, f"say:   {w.upper()}")] * max(0, args.reps - count_existing(w, name))
    for _ in range(max(0, args.unknown - count_existing(UNKNOWN, name))):
        r = random.random()
        if r < 0.7:
            trials.append((UNKNOWN, f"say:   {random.choice(FILLER_WORDS).upper()}   (NOT a command)"))
        elif r < 0.85:
            trials.append((UNKNOWN, "stay SILENT"))
        else:
            trials.append((UNKNOWN, "make a NOISE (cough, clap, tap the desk)"))
    random.shuffle(trials)

    # ---- quick mic check ------------------------------------------------------
    print(f"\nHi {name}! Microphone: {sd.query_devices(kind='input')['name']}")
    input("Mic check: press Enter, say 'hello', then press Enter again... ")
    peak = float(np.abs(record_until_enter()).max() or 0.0)
    print(f"   level = {peak:.2f}  ", end="")
    if peak < 0.05:
        print("-> very quiet. Get closer or raise the input volume in your system settings.")
    elif peak > 0.95:
        print("-> too loud / clipping. Move back or lower the input volume.")
    else:
        print("-> good.")

    if not trials:
        print("\nAll clips already recorded.")
    else:
        print(f"\n{len(trials)} clips to record.")
        print("For each clip: press Enter, say the word ONCE (naturally, like a")
        print("command to the robot), then press Enter again.")
        print("Type  r + Enter  instead of just Enter to redo the previous clip.\n")

    last_path = None
    k = 0
    while k < len(trials):
        label, prompt = trials[k]
        cmd = input(f"[{k + 1}/{len(trials)}]  {prompt:45s}  (Enter to start) ").strip().lower()
        if cmd == "r" and last_path:
            os.remove(last_path)
            last_path = None
            k -= 1
            print(f"   previous clip deleted - redo it:  {trials[k][1]}")
            continue
        audio = record_until_enter()
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if len(audio) < MIN_SECONDS * SR:
            print("   too short - press Enter, speak, THEN press Enter. Try again.")
            continue
        if "SILENT" not in prompt:
            if peak < 0.02:
                print("   too quiet, try again.")
                continue
            if peak > 0.99:
                print("   clipped (too loud), try again.")
                continue
        last_path = save(label, name, audio)
        print(f"   ok ({len(audio) / SR:.1f} s)")
        k += 1

    # ---- zip everything for sending ------------------------------------------
    zip_base = f"recordings_{name}"
    tmp = f"_zip_{name}"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    for label in WORDS + [UNKNOWN]:
        src = os.path.join(OUT_DIR, label)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(tmp, OUT_DIR, label)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if f.startswith(name + "_nohash_"):
                shutil.copy(os.path.join(src, f), dst)
    shutil.make_archive(zip_base, "zip", tmp)
    shutil.rmtree(tmp)
    print(f"\nDone! Send this file:  {os.path.abspath(zip_base + '.zip')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where you left off.")