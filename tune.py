"""Grid-search post-processing on validation probs. Usage: tune.py dir [dir ...] (probs are averaged)."""
import itertools
import json
import sys

import numpy as np

import fil
import post

dirs = sys.argv[1:]
probs, files = post.mean_probs(dirs, "val"), json.load(open(f"{dirs[0]}/val_files.json"))
img_file, anns, by_file = fil.load_train()
gt = {f: [fil.instance_rles(anns[i]) for i in by_file[f]] for f in files}


def score(per_img):
    s = tp = fp = fn = 0
    for f, rl in per_img.items():
        for g in gt[f]:
            a, b, c, d = fil.pq_counts(g, rl)
            s, tp, fp, fn = s + a, tp + b, fp + c, fn + d
    return s / (tp + 0.5 * fp + 0.5 * fn + 1e-9)


res = []
for thr, dil in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7], [1, 5, 9, 15]):
    comps = {f: post.components(p, thr, dil) for f, p in zip(files, probs)}
    for ma, mm in itertools.product([30, 80, 150, 300], [0, 0.7, 0.85, 0.95]):
        res.append((score({f: post.keep(c, ma, mm) for f, c in comps.items()}), thr, dil, ma, mm))
    print("done thr", thr, "dil", dil, "best so far", max(res), flush=True)
res.sort(reverse=True)
print("TOP10", *res[:10], sep="\n")
print("default (thr .6 dil 9 area 50 min 0):", [r for r in res if r[1:] == (0.6, 9, 30, 0)])
