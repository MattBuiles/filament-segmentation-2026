"""U-Net filament segmentation. Self-contained so it runs as a Kaggle script kernel.

Kernels have no internet: the U-Net is built on torchvision resnet34 and ImageNet weights come from a Kaggle dataset.
Env: FOLD (0-4 holdout by contiguous date block, -1 = train on all), EPOCHS, RES, CROP, BS, SEED, SMOKE.
Outputs (OUT): model.pt, val_probs.npy + val_files.json (if FOLD>=0), test_probs.npy + test_files.json.
"""
import glob
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pycocotools import mask as mu

E = os.environ.get
SMOKE = E("SMOKE") == "1"
FOLD, EPOCHS = int(E("FOLD", "2")), int(E("EPOCHS", "40"))
RES, CROP, BS = int(E("RES", "1024")), int(E("CROP", "640")), int(E("BS", "4"))
SEED = int(E("SEED", "0"))
FLAT, LOSS, TGT = E("FLAT") == "1", E("LOSS", "dice"), E("TGT", "soft")
ARCH = E("ARCH", "resnet34")  # resnet34 = torchvision U-Net below; anything else is a segmentation_models_pytorch Unet encoder name
BUDGET = float(E("BUDGET", "0"))  # seconds of training before stopping early; 0 = no limit
OUT = E("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "out")
if SMOKE:
    RES, CROP, BS, EPOCHS = 256, 192, 2, 2
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
if dev == "cpu":
    torch.set_num_threads(2)

ROOT = (glob.glob("/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026", recursive=True) or glob.glob("data/raw/MAGFiLO_1.0_Kaggle_2026"))[0]


def load_xy():
    d = json.load(open(glob.glob(f"{ROOT}/train/*.json")[0]))
    fname = {i["id"]: i["file_name"] for i in d["images"]}
    per = defaultdict(lambda: defaultdict(list))  # file -> annotator image id -> polygons
    for a in d["annotations"]:
        per[fname[a["image_id"]]][a["image_id"]].append(a["segmentation"])
    files = sorted(per)
    if SMOKE:
        files = files[:: max(1, len(files) // 12)][:12]
    X, Y = [], []
    for f in files:
        img = cv2.imread(f"{ROOT}/train/train_images/{f}", cv2.IMREAD_GRAYSCALE)
        views = []
        for polys in per[f].values():  # one binary mask per annotator view
            m = np.zeros((2048, 2048), np.float32)
            for p in polys:
                m = np.maximum(m, mu.decode(mu.merge(mu.frPyObjects(p, 2048, 2048))))
            views.append(m)
        if TGT == "view":  # train on a random single annotator view each time (no consensus)
            X.append(prep(img))
            Y.append(np.stack([cv2.resize(m, (RES, RES), interpolation=cv2.INTER_AREA) for m in views]).astype(np.float16))
            continue
        t = np.mean(views, 0)  # soft target = fraction of annotators marking the pixel
        if TGT == "maj":
            t = (t >= 0.5).astype(np.float32)
        X.append(prep(img))
        Y.append(cv2.resize(t, (RES, RES), interpolation=cv2.INTER_AREA).astype(np.float16))
    return files, X, Y


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, wpath=None):
        super().__init__()
        r = torchvision.models.resnet34()
        if wpath:
            r.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        self.l0, self.pool = nn.Sequential(r.conv1, r.bn1, r.relu), r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.d4, self.d3, self.d2, self.d1, self.d0 = blk(768, 256), blk(384, 128), blk(192, 64), blk(128, 32), blk(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x0 = self.l0(x)
        x1 = self.l1(self.pool(x0)); x2 = self.l2(x1); x3 = self.l3(x2); x4 = self.l4(x3)
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[2:], mode="bilinear", align_corners=False), b], 1)
        d = self.d4(up(x4, x3)); d = self.d3(up(d, x2)); d = self.d2(up(d, x1)); d = self.d1(up(d, x0))
        return self.head(self.d0(F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)))


def prep(img):
    img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_AREA)
    if FLAT:  # divide out limb darkening / large-scale illumination with a heavy blur of the image itself
        f = img.astype(np.float32) + 1
        img = np.clip(f / cv2.GaussianBlur(f, (0, 0), RES / 16), 0, 3)
    return img


def norm(x):
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def augment(img, tgt):
    ang, sc = rng.uniform(-180, 180), rng.uniform(0.8, 1.25)
    cx, cy = rng.uniform(CROP / 2, RES - CROP / 2, 2) if RES > CROP else (RES / 2, RES / 2)
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), ang, sc)
    M[:, 2] += CROP / 2 - np.array([cx, cy])
    kw = dict(dsize=(CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img, tgt = cv2.warpAffine(img, M, **kw), cv2.warpAffine(tgt.astype(np.float32), M, **kw)
    if rng.random() < 0.5:
        img, tgt = img[:, ::-1], tgt[:, ::-1]
    img = norm(img) * rng.uniform(0.8, 1.2) + rng.uniform(-0.2, 0.2)
    return np.ascontiguousarray(img), np.ascontiguousarray(tgt)


def to_in(x):  # (B,H,W) -> (B,3,H,W) grayscale replicated for ImageNet encoders
    return x.unsqueeze(1).repeat(1, 3, 1, 1)


def loss_fn(lg, t):
    bce = F.binary_cross_entropy_with_logits(lg, t, pos_weight=torch.tensor(4.0, device=lg.device))
    p = torch.sigmoid(lg)
    if LOSS == "tversky":  # weight FP 0.7 / FN 0.3: PQ punishes false positives harder than misses
        tp = (p * t).sum()
        return bce + 1 - (tp + 1) / (tp + 0.7 * (p * (1 - t)).sum() + 0.3 * ((1 - p) * t).sum() + 1)
    return bce + 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)


@torch.no_grad()
def predict(model, imgs):
    """4-flip TTA, returns sigmoid probs float16 (N,RES,RES)."""
    model.eval()
    out = []
    for im in imgs:
        x = to_in(torch.from_numpy(norm(im))[None]).to(dev)
        acc = 0
        for fl in ([], [3], [2], [2, 3]):
            xi = x.flip(fl) if fl else x
            with torch.autocast(dev, enabled=dev == "cuda"):
                p = torch.sigmoid(model(xi).float())
            acc = acc + (p.flip(fl) if fl else p)
        out.append((acc / 4)[0, 0].cpu().numpy().astype(np.float16))
    return np.stack(out)


def postprocess(p, thr=0.6, dil=9, min_area=50, min_max=0.0):
    """prob map (RES,RES) -> list of RLE dicts at 2048. Components found on a dilated mask, only original pixels kept."""
    p = cv2.resize(p.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_LINEAR)
    m = (p > thr).astype(np.uint8)
    d = cv2.dilate(m, np.ones((dil, dil), np.uint8)) if dil > 1 else m
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    rles = []
    for k in range(1, n):
        x, y, w, h = st[k, :4]
        sub = (lab[y : y + h, x : x + w] == k) & (m[y : y + h, x : x + w] > 0)
        if sub.sum() < min_area or p[y : y + h, x : x + w][sub].max() < min_max:
            continue
        full = np.zeros((2048, 2048), np.uint8)
        full[y : y + h, x : x + w] = sub
        rles.append(mu.encode(np.asfortranarray(full)))
    return rles


def val_pq(files, probs, per_gt):
    s = tp = fp = fn = 0
    for f, p in zip(files, probs):
        pr = postprocess(p)
        for gt in per_gt[f]:
            if not gt:
                fp += len(pr)
            elif not pr:
                fn += len(gt)
            else:
                iou = np.asarray(mu.iou(gt, pr, [0] * len(pr)))
                hit = iou > 0.5
                s += iou[hit].sum(); tp += hit.sum(); fp += (hit.sum(0) == 0).sum(); fn += (hit.sum(1) == 0).sum()
    return s / (tp + 0.5 * fp + 0.5 * fn + 1e-9)


def main():
    files, X, Y = load_xy()
    n = len(files)
    fold_of = [i * 5 // n for i in range(n)]  # contiguous date blocks: neighbouring observations never straddle the split
    tr = [i for i in range(n) if FOLD < 0 or fold_of[i] != FOLD]
    va = [i for i in range(n) if FOLD >= 0 and fold_of[i] == FOLD]
    print(f"{n} files, train {len(tr)}, val {len(va)}, dev {dev}", flush=True)
    per_gt = {}
    if va:
        d = json.load(open(glob.glob(f"{ROOT}/train/*.json")[0]))
        fname = {i["id"]: i["file_name"] for i in d["images"]}
        g = defaultdict(lambda: defaultdict(list))
        for a in d["annotations"]:
            g[fname[a["image_id"]]][a["image_id"]].append(mu.merge(mu.frPyObjects(a["segmentation"], 2048, 2048)))
        per_gt = {files[i]: list(g[files[i]].values()) for i in va}

    if ARCH == "resnet34":
        w = glob.glob("/kaggle/input/**/resnet34*.pth", recursive=True) + glob.glob("weights/resnet34*.pth")
        model = UNet(None if SMOKE or not w else w[0]).to(dev)
        print("pretrained:", None if SMOKE or not w else w[0], flush=True)
    else:  # needs internet for the ImageNet weights (timm / torchvision hubs)
        import segmentation_models_pytorch as smp
        model = smp.Unet(ARCH, encoder_weights=None if SMOKE else "imagenet", in_channels=3, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = EPOCHS * (len(tr) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")  # fp16 diverged on some seeds in other repos -> scaler + clipping
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(tr)
        tot = 0
        for b in range(len(tr) // BS):
            ib, tb = zip(*[augment(X[i], Y[i][rng.integers(len(Y[i]))] if TGT == "view" else Y[i]) for i in order[b * BS : (b + 1) * BS]])
            x, t = to_in(torch.from_numpy(np.stack(ib))).to(dev), torch.from_numpy(np.stack(tb))[:, None].to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                lg = model(x)
            loss = loss_fn(lg.float(), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        msg = f"ep {ep + 1}/{EPOCHS} loss {tot / (len(tr) // BS):.4f} t {time.time() - t0:.0f}s"
        if va and ((ep + 1) % 10 == 0 or ep + 1 == EPOCHS):
            msg += f" valPQ {val_pq([files[i] for i in va], predict(model, [X[i] for i in va]), per_gt):.4f}"
        print(msg, flush=True)
        if (ep + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{OUT}/model.pt")
        if BUDGET and time.time() - t0 > BUDGET:  # stop early but still write model + predictions
            print("time budget reached", flush=True)
            break

    torch.save(model.state_dict(), f"{OUT}/model.pt")
    if va:
        np.save(f"{OUT}/val_probs.npy", predict(model, [X[i] for i in va]))
        json.dump([files[i] for i in va], open(f"{OUT}/val_files.json", "w"))
    tf = sorted(os.listdir(f"{ROOT}/test/test_images"))[:3 if SMOKE else None]
    timgs = [prep(cv2.imread(f"{ROOT}/test/test_images/{f}", cv2.IMREAD_GRAYSCALE)) for f in tf]
    np.save(f"{OUT}/test_probs.npy", predict(model, timgs))
    json.dump(tf, open(f"{OUT}/test_files.json", "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
