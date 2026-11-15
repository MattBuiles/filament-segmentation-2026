# Solar Filament Segmentation Challenge 2026

U-Net trained on MAGFiLO H-alpha images, with instance post-processing for the Panoptic Quality metric. Final model:
average of three ConvNeXt-Tiny encoders trained on all data. Entry for
https://www.kaggle.com/competitions/filament-segmentation-2026.

## Pipeline
1. `train.py` - grayscale image at 1024 px, replicated to 3 channels; target = fraction of annotators marking each
   pixel (polygons rasterised with pycocotools, as in the scorer); BCE + soft Dice; random rotation/scale/flip/intensity
   crops; AdamW + one-cycle; fp16 with grad clipping. Encoder chosen with `ARCH` (default: torchvision ResNet34;
   `tu-convnext_tiny` for the final models, via segmentation_models_pytorch). Validation = contiguous date block
   (fold 2) so neighbouring observations never straddle the split.
2. Inference (`predict.py`) - 4-flip test-time augmentation, average over the networks of the ensemble.
3. Post-processing - threshold, connected components on a dilated mask (original pixels kept), minimum area and
   minimum peak/mean probability filters; each component is one filament and one RLE row. Parameters tuned on
   validation PQ with `tune.py` / `tune2.py` / `quick.py`.
4. `fil.py` - COCO loading and a PQ scorer identical to the official self-evaluation notebook (IoU > 0.5).
5. `pipeline.ipynb` - executed notebook of the whole pipeline.

## Reproduce the submission (no training needed)
```
pip install -r requirements.txt
python predict.py --images MAGFiLO_1.0_Kaggle_2026/test/test_images --config models/ensemble.json --out submission.csv
```
`models/ensemble.json` is the final ConvNeXt-Tiny ensemble (fp16 weights `cnall.pt`, `cn11.pt`, `cn12.pt`).
`models/ensemble_resnet34_E3.json` reproduces an earlier ResNet34 ensemble (public 0.37). `models/cnf2.pt` is the
ConvNeXt network that never saw fold 2, used by the validation check in the notebook. Training ran as Kaggle script
kernels (`kern/build_pair.py` builds them) on T4 GPUs; PyTorch there was 2.11, locally the versions in `requirements.txt`.

## Results
- Public leaderboard PQ: 0.39 for the three-network ConvNeXt-Tiny average (0.37 for one network; ResNet34 ensemble
  0.37; first ResNet34 baseline with untuned post-processing 0.30).
- Validation (fold 2 held out, PQ over every annotator image): 0.454 for a single ConvNeXt-Tiny with the final
  post-processing; optimistic because the post-processing was tuned on the same images. Two annotators scored against
  each other give 0.345.
- Variants that did not help (illumination flattening, 1536 px, Tversky loss, majority-vote targets, ResNet50, model
  averaging on validation) and all numbers are in `report/report.tex`.
