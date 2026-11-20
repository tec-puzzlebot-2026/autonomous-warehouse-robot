"""
threshold_curve.py - how many real commands vs. non-commands get through, for
different rejection thresholds, using the CURRENT models.pkl and the test
speakers from config.py. Takes seconds (features come from the cache).

Run from the voice/ folder:

    python3 -u scripts/threshold_curve.py               # print the table
    python3 -u scripts/threshold_curve.py --set 0.10    # store the threshold for 10% false accepts
                                                        # in models.pkl (no retraining needed)

Columns:
    false acc.  share of non-command clips accepted as a command
    accepted    share of correctly recognized command clips that pass the threshold
    end-to-end  share of ALL command clips that are recognized correctly AND pass
                (= what you would see on the robot)
"team" columns only use your own test speakers' clips (no Google 'stop').
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import train as T
from common import score_utterance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=float, default=None,
                    help="false-accept rate whose threshold is written into models.pkl")
    args = ap.parse_args()

    bundle = pickle.load(open(C.MODEL_PATH, "rb"))
    print(f"Model: {C.MODEL_PATH}  (current threshold {bundle['threshold']:.3f})")
    print(f"Test speakers: {C.TEST_SPEAKERS or 'automatic'}   USE_ALL_FOR_FINAL={C.USE_ALL_FOR_FINAL}")
    if C.USE_ALL_FOR_FINAL:
        print("[!] USE_ALL_FOR_FINAL=True: the test clips were also trained on -> optimistic numbers")

    print("\nTest files:")
    tests = {name: T.split_class(name)[1] for name in C.WORDS + [C.UNKNOWN]}
    feats = T.extract_all(sorted({p for v in tests.values() for p in v}))

    pos, team = [], []          # (correct?, margin) for every command clip
    per_word = {}
    for i, w in enumerate(C.WORDS):
        for p in tests[w]:
            wl, gl = score_utterance(bundle, feats[p])
            best = int(np.argmax(wl))
            item = (best == i, float(wl[best] - gl))
            pos.append(item)
            per_word.setdefault(w, []).append(item)
            if not T.is_google(p):
                team.append(item)
    neg = np.array([float(np.max(score_utterance(bundle, feats[p])[0])
                          - score_utterance(bundle, feats[p])[1]) for p in tests[C.UNKNOWN]])

    def rates(items, thr):
        if not items:
            return 0.0, 0.0
        corr = np.array([c for c, _ in items])
        marg = np.array([m for _, m in items])
        acc = (marg[corr] > thr).mean() if corr.any() else 0.0
        e2e = ((marg > thr) & corr).mean()
        return acc, e2e

    print(f"\n{'false acc.':>10s} {'threshold':>9s} | {'accepted':>8s} {'end-to-end':>10s} | "
          f"{'team acc.':>9s} {'team e2e':>8s}")
    for fa in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        thr = float(np.percentile(neg, 100 * (1 - fa)))
        a, e = rates(pos, thr)
        ta, te = rates(team, thr)
        print(f"{(neg > thr).mean():>10.1%} {thr:>9.3f} | {a:>8.1%} {e:>10.1%} | {ta:>9.1%} {te:>8.1%}")

    thr10 = float(np.percentile(neg, 90))
    print("\nEnd-to-end per word at 10% false accepts:")
    for w in C.WORDS:
        _, e = rates(per_word.get(w, []), thr10)
        print(f"  {w:11s} {e:6.1%}  ({len(per_word.get(w, []))} clips)")

    if args.set is not None:
        thr = float(np.percentile(neg, 100 * (1 - args.set)))
        bundle["threshold"] = thr
        with open(C.MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)
        print(f"\nSaved threshold {thr:.3f} ({args.set:.0%} false accepts) into {C.MODEL_PATH}")


if __name__ == "__main__":
    main()
