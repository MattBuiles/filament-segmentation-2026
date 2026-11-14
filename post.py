"""Post-processing: probability maps -> instance RLEs, with tuning on validation PQ and submission writing."""
import json
import sys

import cv2
import numpy as np
from pycocotools import mask as mu

cv2.setNumThreads(2)


def components(p, thr, dil):
    """Instances as (rle, area, max_prob). Components come from a dilated mask; only original pixels are kept."""
    p = cv2.resize(p.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_LINEAR)
    m = (p > thr).astype(np.uint8)
    d = cv2.dilate(m, np.ones((dil, dil), np.uint8)) if dil > 1 else m
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    out = []
    for k in range(1, n):
        x, y, w, h = st[k, :4]
        sub = (lab[y : y + h, x : x + w] == k) & (m[y : y + h, x : x + w] > 0)
        a = int(sub.sum())
        if not a:
            continue
        full = np.zeros((2048, 2048), np.uint8)
        full[y : y + h, x : x + w] = sub
        out.append((mu.encode(np.asfortranarray(full)), a, float(p[y : y + h, x : x + w][sub].max())))
    return out


def keep(comps, min_area, min_max):
    return [c[0] for c in comps if c[1] >= min_area and c[2] >= min_max]


def mean_probs(dirs, name):
    return np.mean([np.load(f"{d}/{name}_probs.npy").astype(np.float32) for d in dirs], axis=0)


def write_submission(dirs, path, thr, dil, min_area, min_max):
    probs, files = mean_probs(dirs, "test"), json.load(open(f"{dirs[0]}/test_files.json"))
    rows = ["filament_id,segmentation_rle"]
    for f, p in zip(files, probs):
        for i, r in enumerate(keep(components(p, thr, dil), min_area, min_max), 1):
            rows.append(f"{f.rsplit('.', 1)[0]}_{i},{r['counts'].decode()}")
    open(path, "w").write("\n".join(rows) + "\n")
    print(path, len(rows) - 1, "instances over", len(files), "images")


if __name__ == "__main__":  # python post.py out.csv thr dil min_area min_max dir [dir ...]
    out, thr, dil, ma, mm, *dirs = sys.argv[1:]
    write_submission(dirs, out, float(thr), int(dil), int(ma), float(mm))
