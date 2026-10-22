"""
recognize.py - live recognition logic + a terminal test (no ROS needed).

Usage (from this folder):
    python3 recognize.py            # prints only recognised commands
    python3 recognize.py --verbose  # also prints rejected sounds + scores

How it works:
  mic -> 20 ms blocks -> energy-based Voice Activity Detection (VAD) cuts out
  one utterance -> features -> score with all 11 HMMs -> the word, or
  nothing if the garbage model wins / the margin is too small.

The StreamingRecognizer class is also used by the ROS 2 node (voice_node.py).
"""
import argparse
import collections
import pickle
import queue

import numpy as np

try:                       # imported inside the ROS 2 package
    from . import config as C
    from .common import extract_features, score_utterance, decide
except ImportError:        # run as a plain script from this folder
    import config as C
    from common import extract_features, score_utterance, decide


def db(block):
    return 20 * np.log10(np.sqrt(np.mean(block ** 2)) + 1e-10)


class StreamingRecognizer:
    """Feed it 20 ms audio blocks; it returns a result when an utterance ends.

    result = dict(word=<str or None>, margin=<float>, closest=<str>)
    word is None when the utterance was rejected (not one of the commands).
    """

    def __init__(self, model_path=None, threshold=None):
        with open(model_path or C.MODEL_PATH, "rb") as f:
            self.bundle = pickle.load(f)
        self.words = self.bundle["words"]
        self.threshold = self.bundle["threshold"] if threshold is None else threshold
        self.block = int(C.BLOCK_SECONDS * C.SR)
        self.max_blocks = int(C.MAX_UTTERANCE_SECONDS / C.BLOCK_SECONDS)
        self.floor = None
        self._calib = []
        self._reset()

    def _reset(self):
        self.pre_roll = collections.deque(maxlen=C.PRE_ROLL_BLOCKS)
        self.triggered, self.speech_run, self.silence_run, self.buf = False, 0, 0, []

    @property
    def calibrated(self):
        return self.floor is not None

    def process_block(self, b):
        level = db(b)

        # First second: measure the background noise level.
        if self.floor is None:
            self._calib.append(level)
            if len(self._calib) >= int(1 / C.BLOCK_SECONDS):
                self.floor = float(np.median(self._calib))
            return None

        is_speech = level > self.floor + C.VAD_DB

        if not self.triggered:
            self.pre_roll.append(b)
            self.speech_run = self.speech_run + 1 if is_speech else 0
            if not is_speech:                      # slowly track noise changes
                self.floor = 0.98 * self.floor + 0.02 * level
            if self.speech_run >= C.START_BLOCKS:
                self.triggered, self.buf, self.silence_run = True, list(self.pre_roll), 0
            return None

        self.buf.append(b)
        self.silence_run = 0 if is_speech else self.silence_run + 1
        if self.silence_run < C.END_BLOCKS and len(self.buf) < self.max_blocks:
            return None

        y = np.concatenate(self.buf)
        self._reset()
        return self.classify(y)

    def classify(self, y):
        X = extract_features(y)
        word_ll, garbage_ll = score_utterance(self.bundle, X)
        idx, margin = decide(word_ll, garbage_ll, self.threshold)
        closest = self.words[int(np.argmax(word_ll))]
        return {"word": self.words[idx] if idx is not None else None,
                "margin": margin, "closest": closest}


def main():
    import sounddevice as sd

    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold chosen in training (higher = stricter)")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    args = ap.parse_args()

    rec = StreamingRecognizer(threshold=args.threshold)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=C.SR, channels=1, blocksize=rec.block,
                        dtype="float32", device=args.device, callback=callback):
        print("Calibrating background noise - stay quiet for 1 second...")
        while not rec.calibrated:
            rec.process_block(q.get())
        print(f"Noise floor {rec.floor:.1f} dB. Listening for: {', '.join(rec.words)}")
        while True:
            r = rec.process_block(q.get())
            if r is None:
                continue
            if r["word"]:
                print(f">>> {r['word'].upper()}", flush=True)
                if args.verbose:
                    print(f"    margin {r['margin']:.2f} (threshold {rec.threshold:.2f})")
            elif args.verbose:
                print(f"    (rejected - closest '{r['closest']}', margin {r['margin']:.2f})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
