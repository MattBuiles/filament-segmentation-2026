"""Paired bootstrap of validation PQ (2000 resamples of the 142 held-out observations) for two models at the final post-processing.

Needs the validation probability maps written by train.py (val_probs.npy, val_files.json) in the directories below; run from the repo root.
Produces the intervals quoted in the report (ConvNeXt-Tiny 0.454 [0.439, 0.470], ResNet34 0.442 [0.427, 0.457], paired +0.012)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import fil, post

PP = (0.45, 9, 300, 0.9, 0.85)
dirs = {"convnext_tiny": "kout/pairF/cnf2", "resnet34_80ep": "kout/pairC/e80f2"}
files = json.load(open("kout/pairF/cnf2/val_files.json"))
img_file, anns, by_file = fil.load_train()
gt = {f: [fil.instance_rles(anns[i]) for i in by_file[f]] for f in files}
stats = {}
for name, d in dirs.items():
    probs = np.load(f"{d}/val_probs.npy")
    rows = []
    for f, p in zip(files, probs):
        rl = post.keep(post.components(p, PP[0], PP[1]), PP[2], PP[3], PP[4])
        s = tp = fp = fn = 0
        for g in gt[f]:
            a, b, c, d_ = fil.pq_counts(g, rl); s += a; tp += b; fp += c; fn += d_
        rows.append((s, tp, fp, fn))
    stats[name] = np.array(rows, float)
    print(name, "PQ", round(stats[name][:, 0].sum() / (stats[name][:, 1].sum() + 0.5 * stats[name][:, 2].sum() + 0.5 * stats[name][:, 3].sum()), 4), flush=True)
rng = np.random.default_rng(0)
n = len(files)
pq = lambda a, idx: a[idx, 0].sum() / (a[idx, 1].sum() + 0.5 * a[idx, 2].sum() + 0.5 * a[idx, 3].sum())
res = {k: [] for k in stats}; diff = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    for k in stats: res[k].append(pq(stats[k], idx))
    diff.append(res["convnext_tiny"][-1] - res["resnet34_80ep"][-1])
for k in res: print(k, "95% CI", np.round(np.percentile(res[k], [2.5, 97.5]), 4))
print("paired diff convnext - resnet34: mean", round(np.mean(diff), 4), "95% CI", np.round(np.percentile(diff, [2.5, 97.5]), 4), "P(diff>0)", round(float(np.mean(np.array(diff) > 0)), 3))
