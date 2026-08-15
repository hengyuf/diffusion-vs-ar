"""Aggregate an inference-steps sweep. Usage: _sweep_report.py <sweep_dir> [T,T,T] [seeds]"""
import json
import os
import re
import sys

import numpy as np

d = sys.argv[1]
runs = {}
for name in sorted(os.listdir(d)):
    m = re.fullmatch(r"T(\d+)_s(\d+)", name)
    if not m:
        continue
    p = os.path.join(d, name, "index.json")
    if not os.path.exists(p):
        continue
    T, s = int(m.group(1)), int(m.group(2))
    idx = json.load(open(p))
    runs[(T, s)] = {e["split"]: (e["n_correct"], e["num_samples"]) for e in idx["splits"]}

Ts = sorted({t for t, _ in runs})
seeds = sorted({s for _, s in runs})
splits = [e for e in ["original", "extreme_r0", "extreme_r1_4", "extreme_r5_19",
                      "extreme_r20_49", "extreme_r50_99", "extreme_r100_plus"]
          if any(e in v for v in runs.values())]

print("runs found:", len(runs), " T =", Ts, " seeds =", seeds)
print()
hdr = "%-18s" % "split"
for T in Ts:
    hdr += "  T=%-3d mean (per-seed)      " % T
print(hdr)
print("-" * len(hdr))

means = {}
for sp in splits:
    line = "%-18s" % sp
    for T in Ts:
        vals = [runs[(T, s)][sp][0] / runs[(T, s)][sp][1] for s in seeds if (T, s) in runs]
        means[(sp, T)] = float(np.mean(vals))
        per = "/".join("%.3f" % v for v in vals)
        line += "  %.3f (%s)      " % (np.mean(vals), per)
    print(line)

print()
print("--- hard splits pooled (everything except 'original') ---")
for T in Ts:
    c = tot = 0
    for s in seeds:
        if (T, s) not in runs:
            continue
        for sp in splits:
            if sp == "original":
                continue
            a, b = runs[(T, s)][sp]
            c += a
            tot += b
    print("T=%-4d pooled accuracy %.4f   (%d / %d boards over %d seeds)"
          % (T, c / tot, c, tot, len(seeds)))

print()
print("--- change vs T=20 (mean over seeds) ---")
base = Ts[0]
for sp in splits:
    deltas = "  ".join("T=%d %+.3f" % (T, means[(sp, T)] - means[(sp, base)]) for T in Ts[1:])
    print("%-18s %s" % (sp, deltas))
