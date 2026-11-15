"""Wider post-process grid incl. mean-confidence filter. Usage: tune2.py dir [dir ...]"""
import itertools, json, sys
import fil, post

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
for thr, dil in itertools.product([0.4, 0.5], [9, 15, 21]):
    comps = {f: post.components(p, thr, dil) for f, p in zip(files, probs)}
    for ma, mm, mn in itertools.product([150, 300, 500, 800], [0.9, 0.95, 0.98], [0, 0.6, 0.7, 0.8]):
        res.append((round(score({f: post.keep(c, ma, mm, mn) for f, c in comps.items()}), 4), thr, dil, ma, mm, mn))
    print("done", thr, dil, "best", max(res), flush=True)
res.sort(reverse=True)
print("TOP8", *res[:8], sep="\n")
