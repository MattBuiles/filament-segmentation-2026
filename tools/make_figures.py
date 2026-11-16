"""Figures of the report: pipeline diagram and examples. Needs the data folder and the probability maps (val/test) produced by train.py.
Run from the repo root; paths below point at the local Kaggle kernel outputs used for the submission."""
import json, sys
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, ".")
import fil, post
from pycocotools import mask as mu

# ---- pipeline diagram
fig, ax = plt.subplots(figsize=(3.4, 4.3)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 14)
steps = ["H-alpha image\n2048x2048, grayscale", "resize to 1024x1024,\nper-image standardisation",
         "U-Net, ConvNeXt-Tiny encoder\n(5 seeds) x 4-flip TTA\n-> mean probability map", "threshold p > 0.45;\ndilate 9x9 only for connectivity;\nlabel components,\nkeep original pixels",
         "filters: area >= 300 px,\nmax p >= 0.9, mean p >= 0.85", "one RLE per filament\n-> submission CSV"]
ys = [12.6, 10.4, 8.0, 5.3, 3.0, 0.9]
for t, y in zip(steps, ys):
    ax.add_patch(FancyBboxPatch((0.3, y - 0.8), 9.4, 1.6, boxstyle="round,pad=0.05", fc="#eef3fa", ec="#33557f", lw=0.8))
    ax.text(5, y, t, ha="center", va="center", fontsize=5.6)
for y0, y1 in zip(ys[:-1], ys[1:]):
    ax.annotate("", xy=(5, y1 + 0.95), xytext=(5, y0 - 0.9), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#33557f"))
fig.savefig("report/fig_pipeline.pdf", bbox_inches="tight"); plt.close(fig)

# ---- examples
files = json.load(open("kout/pairF/cnf2/val_files.json"))
img_file, anns, by_file = fil.load_train()
vp = np.load("kout/pairF/cnf2/val_probs.npy", mmap_mode="r")
cnt = [(len(anns[by_file[f][0]]), i) for i, f in enumerate(files)]
cnt.sort(); i = cnt[int(len(cnt) * 0.75)][1]; f = files[i]  # an image with a fairly typical-to-busy number of filaments
def overlay(img, rles, seed=0):
    ov = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for k, r in enumerate(rles):
        m = mu.decode(r) if isinstance(r, dict) else r
        ov[m > 0] = (np.array(plt.cm.tab10(k % 10)[:3]) * 255 * 0.8 + ov[m > 0] * 0.2).astype(np.uint8)
    return ov
img = cv2.imread(f"data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images/{f}", cv2.IMREAD_GRAYSCALE)
soft = np.mean([np.max([mu.decode(r) for r in fil.instance_rles(anns[a])], 0) for a in by_file[f]], 0)
pred = [c[0] for c in post.components(np.asarray(vp[i], np.float32), 0.45, 9) if c[1] >= 300 and c[2] >= 0.9 and c[3] >= 0.85]
tf = json.load(open("kout/pairF/cnall/test_files.json"))
j = 24; tname = tf[j]
tp = np.mean([np.asarray(np.load(f"{d}/test_probs.npy", mmap_mode="r")[j], np.float32) for d in ["kout/pairF/cnall", "kout/pairI/cn11", "kout/pairI/cn12", "kout/pairJ/cn13", "kout/pairJ/cn14"]], 0)
timg = cv2.imread(f"data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images/{tname}", cv2.IMREAD_GRAYSCALE)
tpred = [c[0] for c in post.components(tp, 0.45, 9) if c[1] >= 300 and c[2] >= 0.9 and c[3] >= 0.85]
S = 512
sm = lambda a, inter=cv2.INTER_AREA: cv2.resize(a, (S, S), interpolation=inter)
fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.75))
ax[0, 0].imshow(sm(img), cmap="gray"); ax[0, 0].set_title("validation image", fontsize=7)
ax[0, 1].imshow(sm(soft.astype(np.float32)), vmin=0, vmax=1, cmap="viridis"); ax[0, 1].set_title("annotators' agreement (training target)", fontsize=7)
ax[0, 2].imshow(sm(overlay(img, pred), cv2.INTER_AREA)); ax[0, 2].set_title(f"prediction: {len(pred)} filaments", fontsize=7)
ax[1, 0].imshow(sm(timg), cmap="gray"); ax[1, 0].set_title("test image", fontsize=7)
ax[1, 1].imshow(sm(tp.astype(np.float32) if tp.shape[0] == 2048 else cv2.resize(tp, (2048, 2048))), vmin=0, vmax=1, cmap="viridis"); ax[1, 1].set_title("mean probability, 5 networks", fontsize=7)
ax[1, 2].imshow(sm(overlay(timg, tpred))); ax[1, 2].set_title(f"prediction: {len(tpred)} filaments", fontsize=7)
for a in ax.ravel(): a.axis("off")
plt.tight_layout(pad=0.4); fig.savefig("report/fig_examples.png", dpi=170); plt.close(fig)
print("val", f, len(pred), "test", tname, len(tpred))
