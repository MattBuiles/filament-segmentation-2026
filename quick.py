"""Quick val PQ of single models and ensembles at fixed post-process params. Usage: quick.py thr dil area minmax dirA [dirB ...] (also prints each alone)."""
import json, sys
import fil, post
thr, dil, ma, mm, *dirs = sys.argv[1:]; thr, dil, ma, mm = float(thr), int(dil), int(ma), float(mm)
files = json.load(open(f"{dirs[0]}/val_files.json"))
img_file, anns, by_file = fil.load_train()
gt = {f: [fil.instance_rles(anns[i]) for i in by_file[f]] for f in files}
def pq(ds):
    probs = post.mean_probs(ds, "val"); s = tp = fp = fn = 0
    for f, p in zip(files, probs):
        rl = post.keep(post.components(p, thr, dil), ma, mm)
        for g in gt[f]:
            a, b, c, d = fil.pq_counts(g, rl); s, tp, fp, fn = s + a, tp + b, fp + c, fn + d
    return s / (tp + 0.5 * fp + 0.5 * fn)
import os
if not os.environ.get("ONLY"):
    for d in dirs: print(d, round(pq([d]), 4), flush=True)
print("ENSEMBLE", dirs, round(pq(dirs), 4), flush=True)
