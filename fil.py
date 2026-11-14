"""Shared helpers: COCO loading, GT rasterisation, PQ scorer, RLE encoding."""
import glob
import json
from collections import defaultdict

import numpy as np
from pycocotools import mask as mu

SIZE = 2048
ROOT = glob.glob("data/raw/*")[0] if glob.glob("data/raw/*") else "data/raw/MAGFiLO_1.0_Kaggle_2026"


def load_train(root=ROOT):
    d = json.load(open(glob.glob(f"{root}/train/*.json")[0]))
    img_file = {i["id"]: i["file_name"] for i in d["images"]}
    anns = defaultdict(list)  # annotated-image id -> polygons
    for a in d["annotations"]:
        anns[a["image_id"]].append(a["segmentation"])
    by_file = defaultdict(list)  # observation file -> annotated-image ids (one per annotator)
    for iid, f in img_file.items():
        by_file[f].append(iid)
    return img_file, anns, by_file


def poly_to_rle(poly):
    # pycocotools rasterisation: the scorer's GT is built this way, cv2.fillPoly differs on edges
    return mu.merge(mu.frPyObjects(poly, SIZE, SIZE))


def instance_rles(polys):
    return [poly_to_rle(p) for p in polys]


def encode_rle(mask):
    # column-major (Fortran) as pycocotools requires; row-major silently transposes the mask
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode()


def decode_rle(counts):
    return mu.decode({"size": [SIZE, SIZE], "counts": counts.encode()})


def pq_counts(gt, pr):
    """gt, pr: lists of RLE dicts. (sum IoU of TPs, n_tp, n_fp, n_fn); official rule, IoU>0.5."""
    if not gt:
        return 0.0, 0, len(pr), 0
    if not pr:
        return 0.0, 0, 0, len(gt)
    iou = np.asarray(mu.iou(gt, pr, [0] * len(pr)))  # (n_gt, n_pred), computed on RLE, no dense masks
    hit = iou > 0.5
    return float(iou[hit].sum()), int(hit.sum()), int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum())


def pq(pairs):
    """pairs: iterable of (gt_rles, pred_rles); pooled over all images."""
    s = tp = fp = fn = 0
    for gt, pr in pairs:
        a, b, c, d = pq_counts(gt, pr)
        s, tp, fp, fn = s + a, tp + b, fp + c, fn + d
    den = tp + 0.5 * fp + 0.5 * fn
    return s / den if den else 0.0
