"""Reproduce the submission: ensemble inference (4-flip TTA) + instance post-processing -> CSV.

python predict.py --images MAGFiLO_1.0_Kaggle_2026/test/test_images --config models/ensemble.json --out submission.csv
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pycocotools import mask as mu


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    """U-Net with an ImageNet ResNet34 encoder (same definition as train.py)."""

    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet34()
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


def build(arch=None):
    """None = the torchvision ResNet34 U-Net above; otherwise a segmentation_models_pytorch Unet encoder name."""
    if arch is None:
        return UNet()
    import segmentation_models_pytorch as smp
    return smp.Unet(arch, encoder_weights=None, in_channels=3, classes=1)


def prep(img, res, flat):
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
    if flat:  # divide out limb darkening with a heavy blur of the image itself
        f = img.astype(np.float32) + 1
        img = np.clip(f / cv2.GaussianBlur(f, (0, 0), res / 16), 0, 3)
    x = img.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


@torch.no_grad()
def predict(model, x, dev):
    x = torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1).to(dev)
    acc = 0
    for fl in ([], [3], [2], [2, 3]):
        xi = x.flip(fl) if fl else x
        with torch.autocast(dev, enabled=dev == "cuda"):
            p = torch.sigmoid(model(xi).float())
        acc = acc + (p.flip(fl) if fl else p)
    return (acc / 4)[0, 0].cpu().numpy()


def instances(p, thr, dil, min_area, min_max, min_mean=0.0):
    """Components of the thresholded map found on a dilated mask; only original pixels are kept, as RLE at 2048x2048."""
    p = cv2.resize(p, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    m = (p > thr).astype(np.uint8)
    d = cv2.dilate(m, np.ones((dil, dil), np.uint8)) if dil > 1 else m
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    out = []
    for k in range(1, n):
        x, y, w, h = st[k, :4]
        sub = (lab[y : y + h, x : x + w] == k) & (m[y : y + h, x : x + w] > 0)
        v = p[y : y + h, x : x + w][sub]
        if sub.sum() < min_area or v.max() < min_max or v.mean() < min_mean:
            continue
        full = np.zeros((2048, 2048), np.uint8)
        full[y : y + h, x : x + w] = sub
        out.append(mu.encode(np.asfortranarray(full))["counts"].decode())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--config", default="models/ensemble.json")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (for quick checks)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    base = os.path.dirname(os.path.abspath(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for m in cfg["models"]:
        net = build(m.get("arch")).to(dev).eval()
        net.load_state_dict({k: v.float() for k, v in torch.load(f"{base}/{m['file']}", map_location=dev, weights_only=True).items()})
        models.append((net, m["res"], m.get("flat", False)))
    files = sorted(os.listdir(a.images))[: a.limit or None]
    pp = cfg["postprocess"]
    rows = ["filament_id,segmentation_rle"]
    for f in files:
        img = cv2.imread(f"{a.images}/{f}", cv2.IMREAD_GRAYSCALE)
        probs = []
        for net, res, flat in models:  # every model's map is brought to 1024 before averaging
            p = predict(net, prep(img, res, flat), dev)
            probs.append(cv2.resize(p, (1024, 1024), interpolation=cv2.INTER_AREA) if res != 1024 else p)
        for i, r in enumerate(instances(np.mean(probs, 0), **pp), 1):
            rows.append(f"{f.rsplit('.', 1)[0]}_{i},{r}")
        print(f, flush=True)
    open(a.out, "w").write("\n".join(rows) + "\n")
    print(a.out, len(rows) - 1, "instances")


if __name__ == "__main__":
    main()
