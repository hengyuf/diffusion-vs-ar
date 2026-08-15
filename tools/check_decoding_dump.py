#!/usr/bin/env python
"""Sanity-check a decoding-path dump produced by tools/dump_decoding_paths.py.

Verifies the dump against a reference accuracy table (if given), checks that the
recorded probabilities agree with the recorded boards, and reports the commit /
re-mask dynamics of the top-k decoder.

  python tools/check_decoding_dump.py output/decoding_paths/checkpoint-45500 [split]
"""
import base64
import json
import os
import sys

import numpy as np

# accuracies logged by the Trainer at step 45500 for this checkpoint, as a regression
# reference. Decoding is stochastic (Gumbel noise in the top-k remask), so these are
# not expected to reproduce exactly -- only to track.
REF = {"original": 1.0, "extreme_r0": 0.705, "extreme_r1_4": 0.39, "extreme_r5_19": 0.05,
       "extreme_r20_49": 0.045, "extreme_r50_99": 0.035, "extreme_r100_plus": 0.035}

d = sys.argv[1]
idx = json.load(open(os.path.join(d, "index.json")))

print("%-18s %6s %9s %11s" % ("split", "n", "acc", "ref@45500"))
for s in idx["splits"]:
    ref = REF.get(s["split"])
    print("%-18s %6d %9.3f %11s" % (s["split"], s["num_samples"], s["accuracy"],
                                    "%.3f" % ref if ref is not None else "-"))

name = sys.argv[2] if len(sys.argv) > 2 else idx["splits"][-1]["split"]
data = json.load(open(os.path.join(d, name + ".json")))
T = data["diffusion_steps"]
n = len(data["samples"])
print("\n--- consistency checks on %s (%d samples) ---" % (name, n))

bad_shape = bad_argmax = bad_final = bad_correct = ties = 0
mass_lo = 1.0
for s in data["samples"]:
    probs = np.frombuffer(base64.b64decode(s["probs_b64"]), dtype=np.uint8)
    if probs.size != T * 81 * 9:
        bad_shape += 1
        continue
    probs = probs.reshape(T, 81, 9)
    mass_lo = min(mass_lo, (probs.sum(2) / 255.0).min())
    # the recorded argmax digit must be the board the model emitted. Probabilities are
    # stored as uint8, so two digits can round to the same byte -- that is a display
    # artifact, not a disagreement, and is counted separately.
    for k in range(T):
        am = probs[k].argmax(1) + 1
        for i, ch in enumerate(s["steps"][k]["x0"]):
            if ch.isdigit() and int(ch) != am[i]:
                if ch != "0" and probs[k, i, am[i] - 1] == probs[k, i, int(ch) - 1]:
                    ties += 1
                else:
                    bad_argmax += 1
    if s["steps"][-1]["x0"] != s["pred"]:
        bad_final += 1
    if (s["pred"] == s["solution"]) != s["correct"]:
        bad_correct += 1

cells = n * T * 81
print("probs arrays with wrong shape          :", bad_shape)
print("argmax(probs) != x0, real disagreement :", bad_argmax)
print("argmax(probs) != x0, uint8 tie         : %d  (%.4f%% of %d cells)"
      % (ties, 100.0 * ties / cells, cells))
print("final x0 != pred (samples)             :", bad_final)
print("'correct' flag != (pred==solution)     :", bad_correct)
print("min prob mass on digits 1-9 (any cell) : %.3f" % mass_lo)

# decoder dynamics: does top-k re-mask cells it already committed?
tot_rev = tot_rem = tot_flip = 0
per_rev = [0] * T
per_rem = [0] * T
with_remask = 0
for s in data["samples"]:
    steps, had = s["steps"], False
    for k in range(T):
        cur = steps[k]["xt"]
        nxt = steps[k + 1]["xt"] if k + 1 < T else s["pred"]
        rev = sum(1 for i in range(81) if cur[i] == "." and nxt[i] != ".")
        rem = sum(1 for i in range(81) if cur[i] != "." and nxt[i] == ".")
        tot_flip += sum(1 for i in range(81)
                        if cur[i] != "." and nxt[i] != "." and cur[i] != nxt[i])
        per_rev[k] += rev
        per_rem[k] += rem
        tot_rev += rev
        tot_rem += rem
        had |= rem > 0
    with_remask += had

print("\n--- decoder dynamics on %s ---" % name)
print("newly-committed cell-events :", tot_rev)
print("re-masked cell-events       :", tot_rem)
print("value changed in place      :", tot_flip)
print("samples with >=1 re-mask    : %d/%d" % (with_remask, n))
print("\nstep  avg_committed  avg_remasked")
for k in range(T):
    print("%4d %13.2f %14.2f" % (k + 1, per_rev[k] / n, per_rem[k] / n))
