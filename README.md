# Solar Filament Segmentation Challenge 2026

U-Net (ImageNet ResNet34 encoder) trained on MAGFiLO H-alpha images, with instance post-processing for the
Panoptic Quality metric. Entry for https://www.kaggle.com/competitions/filament-segmentation-2026.

## Pipeline
1. `train.py` – grayscale image at 1024 px, replicated to 3 channels; target = fraction of annotators marking each
   pixel (polygons rasterised with pycocotools, as in the scorer); BCE + soft Dice; random rotation/scale/flip/intensity
   crops; AdamW + one-cycle; fp16 with grad clipping. Validation = contiguous date block (fold 2) so neighbouring
   observations never straddle the split.
2. Inference (`predict.py`) – 4-flip test-time augmentation, optional ensemble average.
3. Post-processing – threshold, connected components on a dilated mask (original pixels kept), minimum area and
   minimum peak probability filters; each component is one filament and one RLE row. Parameters tuned on validation PQ
   with `tune.py` / `quick.py`.
4. `fil.py` – COCO loading and a PQ scorer identical to the official self-evaluation notebook (IoU > 0.5).

## Reproduce the submission (no training needed)
```
pip install -r requirements.txt
python predict.py --images MAGFiLO_1.0_Kaggle_2026/test/test_images --config models/ensemble.json --out submission.csv
```
Weights are in `models/` (fp16). Training ran as Kaggle script kernels (`kern/build_pair.py` builds them) on T4 GPUs;
PyTorch there was 2.11, locally the versions in `requirements.txt`.

## Results (fold-2 validation, PQ over every annotator image)
See `report/` for the numbers of each experiment.
