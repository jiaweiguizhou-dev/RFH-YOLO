# RFH-YOLO

Source release candidate for **RFH-YOLO: Relational-Field Hypergraph Modeling for Batik Pattern Detection**.
This repository contains the local RFH-YOLO implementation built on [Hyper-YOLO](https://github.com/iMoonLab/Hyper-YOLO) and a modified Ultralytics YOLO 8.0.227 source tree. The manuscript source is `My paper.tex`.
Publication, checkpoint distribution, and dataset access remain subject to author approval and applicable permissions. **Existing training logs and checkpoints are included, but no training, evaluation, FPS benchmark, or deployment test was rerun to prepare this README.**

## Layout

- `train.py`: main single-GPU training entry point.
- `test.py`: main test-set evaluation and image prediction entry point, with auxiliary-head pruning and convolution fusion.
- `train_repmanetpp_original.py`, `test_repmanetpp_original.py`: alternate training and evaluation scripts; not identical to the main workflow.
- `ultralytics/`: modified local implementation, including `nn/modules/block.py` (RFHM, RepMANet, RepMixConv), `nn/modules/head.py` (dual-head Detect), `utils/loss.py` (progress-dependent auxiliary supervision), and `nn/tasks.py` (`convert_to_deploy`).
- `ultralytics/cfg/models/hyper-yolo/hyper-yolon-repmanetpp.yaml`: main model configuration; `*-original.yaml` and `hyper-yolon-repmanet.yaml` are alternative configurations, not interchangeable ablation results.
- `runs/train2/args.yaml`, `runs/train2/results.csv`: existing recorded run arguments and training log.
- `runs/train2/weights/best.pt`, `runs/train2/weights/epoch300.pt`: included checkpoints; their performance was not independently revalidated during documentation preparation.
- `environment.yaml`, `requirements.txt`: inherited environment and dependency specifications, not a verified environment lockfile.

The archive does **not** contain `scripts/run.py`, `configs/train/batikdet-recorded.yaml`, `configs/datasets/batikdet.yaml`, `docs/DATA_AND_WEIGHTS.md`, `CITATION.cff`, or `.zenodo.json`. Commands referring to those paths do not apply to this code.

## Environment and installation

The inherited `environment.yaml` records Python 3.8.16, PyTorch 2.0.1, torchvision 0.15.2, and CUDA 11.7-related packages; **it is not a verified RFH-YOLO experiment environment**. In particular, `train.py` calls `torch.serialization.add_safe_globals`, which is unavailable in PyTorch 2.0.1. Choose a supported PyTorch/torchvision/CUDA combination that provides this API and is compatible with your NVIDIA driver; see the [official PyTorch installation guidance](https://pytorch.org/get-started/previous-versions/).

```bash
conda create -n rfh-yolo python=3.10 -y
conda activate rfh-yolo
# Install a compatible PyTorch and torchvision build for your CUDA/driver first.
python -m pip install "numpy<2" -r requirements.txt
python -m pip install -e . --no-deps
python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__file__); print(hasattr(torch.serialization, 'add_safe_globals'))"
```

Run all commands from the repository root. Confirm that `ultralytics.__file__` points to this **local fork**, not an unrelated PyPI installation. These commands have not been runtime-tested for this archive. Record the working environment (`python -m pip freeze`), GPU, driver, and CUDA details for reproducibility. **Load only trusted checkpoints**: the inherited loaders use Python pickle, and `test.py` defaults to `torch.load(..., weights_only=False)` when the caller does not specify that argument.

## BatikDet

**BatikDet images, labels, and the external dataset YAML are not included.** The main scripts reference `/root/sj-tmp/dataset.yaml`; `test.py` also references `/root/sj-tmp/test/images`. Edit these paths to point to your authorized local dataset. Preserve the original train/validation/test split and class IDs; do not generate a new random split when reproducing the paper.

```text
BatikDet/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
```

YOLO labels contain `class_id x_center y_center width height`, using normalized coordinates and zero-based class IDs 0–15. Each image should match the corresponding `.txt` label file by filename stem. This folder layout is illustrative, not a claim about the original server layout.

The manuscript describes **9,777 images, 75,249 annotated instances, and train/val/test splits of 7,820/978/979 images**. The 16 classes, in ID order, are: Bird, Drum, Butterfly, Dragon, Fish, Human, Pomegranate, Vortex, Pomegranate Flower, Lotus, Interlocking Lozenge, Cloud, Meander, Bat, Ox, and Longevity.
Dataset Availability

The BatikDet dataset contains 9,777 images and 75,249 annotated instances across 16 batik pattern categories.
The dataset is available upon reasonable request for academic research purposes. To request access, please contact the corresponding author at wjia@gzu.edu.cn.
Dataset access is subject to applicable data-use agreements and copyright restrictions.
## Train

Edit the `data` path in `train.py` and run:

```bash
python train.py
```

The main script loads `ultralytics/cfg/models/hyper-yolo/hyper-yolon-repmanetpp.yaml`. Its recorded training setup uses **500 planned epochs, batch 20, image size 640, SGD, GPU 0, workers 8, patience 100, deterministic mode, and AMP disabled**. The saved `runs/train2/args.yaml` additionally records `pretrained: false`, `seed: 0`, `lr0: 0.01`, `momentum: 0.937`, `weight_decay: 0.0005`, and box/class/DFL loss weights **7.5/0.5/1.5**. Early stopping can finish before epoch 500. The model YAML's `nc: 80` is a template value, overridden by the 16-class dataset YAML when training.

The architecture includes **RFHM** (global/local relation scales, adaptive membership, residual gating), **RepMANet** (`RepMixConv` training branches `3×3 + 1×1`, fused to a single `3×3` convolution), and **PD-DHAS** (a progress-dependent auxiliary loss with maximum weight **0.25**, warm-up during the first 5%, constant to 50%, then cosine decay). `Detect` receives six features, `[B3, B4, B5, N3, N4, N5]`: three auxiliary and three main features. Inference uses the main branch. The separate `train_repmanetpp_original.py` defaults to five GPUs (`[0,1,2,3,4]`) and another model YAML; adjust its settings before use.

A separately validated true single-head training implementation is not supplied. Setting auxiliary loss weight to zero does **not** remove auxiliary-head computation. Do not use incompatible three-input detection YAMLs with the current six-input `Detect` head.

## Validate and predict

The main `test.py` uses hard-coded settings. Edit them to match your paths before running:

```python
WEIGHTS = "runs/train2/weights/best.pt"
DATA_YAML = "/root/sj-tmp/dataset.yaml"
VAL_IMAGES = "/root/sj-tmp/test/images"
SAVE_ROOT = "runs/RFH_YOLO_Aug/DA5_seed02/weights/"
```

```bash
python test.py
```

The script loads the checkpoint, uses `convert_to_deploy(fuse=False)` to remove auxiliary detection towers, invokes `model.fuse()`, audits `RepMixConv` fusion, validates on the **test split**, and predicts images from `VAL_IMAGES`. Validation uses `imgsz=640`, `batch=20`, and `device=0`. Prediction uses `conf=0.25`, `iou=0.7`, and saves detections, text labels, confidence values, and crops. The validation call does **not** explicitly set `conf`, NMS IoU, or `max_det`; the saved run's `args.yaml` records `conf=0.001`, `iou=0.7`, and `max_det=300`.

`SAVE_ROOT` controls logging and validation output, but is not explicitly passed as the prediction `project`; inspect the actual prediction output directory. Validation timing is **not** the paper's dedicated synchronized batch-1 FPS protocol. No paper accuracy or speed metric has been rerun for this README.

An alternative evaluator accepts explicit paths:

```bash
python test_repmanetpp_original.py --weights runs/train2/weights/best.pt --data /absolute/path/to/dataset.yaml
```

For a **trusted** legacy checkpoint, that alternate script provides `--trusted-checkpoint`. It does not use the same explicit auxiliary-head pruning procedure as `test.py`.

## Deploy

`DetectionModel.convert_to_deploy(fuse=True)` returns an evaluation-mode model copy with auxiliary detection towers removed and supported convolution branches fused. No `scripts/run.py deploy` command or independently validated checkpoint-export workflow exists in this repository.

```python
# Example after loading a trusted checkpoint with ultralytics.YOLO:
deploy_model = yolo_model.model.convert_to_deploy(fuse=True)
```

Keep the original training checkpoint. A physically pruned model state dictionary is not strictly compatible with the original dual-head training graph; do not use a deployment checkpoint to resume training. Verify numerical equivalence, checkpoint reload behavior, and inference performance in your target runtime. ONNX/TensorRT export has not been validated for this RFH-YOLO release.

## Citation and license

Cite the final author-approved RFH-YOLO publication when its bibliographic details are available. Also acknowledge [Hyper-YOLO](https://github.com/iMoonLab/Hyper-YOLO) and [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) as underlying work and software. The original Hyper-YOLO citation appears in its upstream README. This archive contains `My paper.tex` but **no `CITATION.cff` or `.zenodo.json`**; no RFH-YOLO paper DOI, Zenodo DOI, or public repository URL is invented here.

The inherited license is **AGPL-3.0**; retain applicable upstream copyright and license notices. Before public release, review permissions for the included checkpoints and training artifacts, remove hard-coded private server paths as appropriate, and exclude `.release-backup/` (which contains local recovery files). No dataset or checkpoint is relicensed by this README.
