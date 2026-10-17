# RFH-YOLO

Source release candidate for **RFH-YOLO: Relational-Field Hypergraph Modeling for Batik Pattern Detection**.
This fork builds on [Hyper-YOLO](https://github.com/iMoonLab/Hyper-YOLO) and Ultralytics YOLO.
Publication, checkpoint release, and dataset access are separate author-controlled steps. No metrics were rerun during packaging.

## Layout

- `ultralytics/`: preserved local implementation and original model configurations.
- `configs/train/batikdet-recorded.yaml`: recorded training arguments with dataset/output locations made portable.
- `configs/datasets/batikdet.yaml`: 16-class dataset template from the manuscript.
- `scripts/run.py`: portable training, validation, prediction, and deployment entry points.
- `docs/`: data/weights access, reproducibility limitations, release instructions and code availability wording.
- `CITATION.cff`, `.zenodo.json`: citation and draft archive metadata.
- `THIRD_PARTY_NOTICES.md`, `LICENSE`: upstream attribution and inherited AGPL-3.0 license.

## Environment and installation

The inherited environment export records Python 3.8.16, PyTorch 2.0.1, torchvision 0.15.2 and CUDA 11.7 libraries; this is not a verified RFH-YOLO experiment environment. Original `train.py` calls `torch.serialization.add_safe_globals`, which is absent in that old PyTorch version; the portable entry point does not call it.
A candidate setup is Python 3.10, PyTorch 2.0.1, torchvision 0.15.2, CUDA 11.7 on Linux with a compatible NVIDIA driver. CPU is suitable for smoke checks; GPU training is recommended. This candidate has not been installed or runtime-tested during packaging. See the [official version matrix](https://pytorch.org/get-started/previous-versions/).

```bash
conda env create -f environment.yaml
conda activate rfh-yolo
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install "numpy<2" -r requirements.txt
python -m pip install -e . --no-deps
python -c "import ultralytics; print(ultralytics.__file__)"
```

Run all commands at this repository root. Confirm that the imported package is this local fork. Do not replace it with the current upstream PyPI package. `requirements.txt` preserves inherited lower bounds, not an exact lockfile. Export the actual working environment with `python -m pip freeze` and record GPU/driver details before reporting reproduction results.

## BatikDet

No images, annotations or split lists are redistributed. Obtain authorized access from the corresponding author subject to the data-use agreement and rights-holder permissions. See [dataset access](docs/DATA_AND_WEIGHTS.md). Edit `path` in `configs/datasets/batikdet.yaml` to the dataset root; preserve the supplied split membership and class IDs.

```text
BatikDet/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
```

YOLO labels contain `class_id x_center y_center width height` with normalized coordinates and zero-based IDs 0–15. Match each image stem with its `.txt` label. Folder names are a portable template, not evidence of the original on-disk split layout. Do not generate a new random partition for paper reproduction.

## Train

```bash
python scripts/run.py train --data configs/datasets/batikdet.yaml --device 0
```

The recorded recipe uses 500 planned epochs, batch 20, image size 640, SGD, seed 0, deterministic mode, AMP off, patience 100, no external pretraining and workers 8. All remaining values are retained in the recorded configuration. Early stopping can end before 500 epochs. The `*-original.yaml` alternate configuration is retained, but is not certified equivalent to a particular paper ablation. A true single-head training implementation is not supplied by this packaging step: the current `Detect` requires paired auxiliary/main inputs. Setting auxiliary loss to zero is not equivalent to removing the auxiliary computation. Do not use inherited three-input detection YAMLs with this head.

## Validate and predict

Weights are excluded pending author approval and provenance checks. Only load trusted checkpoints; inherited loaders use Python pickle. PyTorch version changes may require checkpoint compatibility work. No checkpoint is downloaded automatically by the wrapper when `--weights` is absent.

```bash
python scripts/run.py val --weights weights/best.pt --data configs/datasets/batikdet.yaml --split test --device 0
python scripts/run.py predict --weights weights/best.pt --source path/to/images --device 0
```

Evaluation uses confidence 0.001, NMS IoU 0.7, max detections 300. Prediction uses confidence 0.25. Prediction timing is not the paper FPS protocol. No reported paper accuracy or speed is asserted to have been reproduced here.

## Deploy

```bash
python scripts/run.py deploy --weights weights/best.pt --output runs/deploy.pt
python scripts/run.py predict --weights runs/deploy.pt --source path/to/images --device 0
```

`DetectionModel.convert_to_deploy(fuse=True)` creates an evaluation copy, removes auxiliary towers and fuses reparameterizable branches. Evaluation already bypasses the auxiliary towers; physical pruning removes their parameters. Deployment checkpoints are inference-only; do not resume training or strictly load a pruned state dictionary into the dual-head training graph. Keep original training weights. Verify numerical equivalence and checkpoint reload in the actual runtime before deployment. ONNX/TensorRT export is not validated.

## Citation and license

Use `CITATION.cff` for software citation; replace draft metadata with the approved version, repository URL and archive DOI when issued. Authors are transcribed from the supplied manuscript and need release approval. No paper DOI or Zenodo DOI is invented. Cite Hyper-YOLO and Ultralytics as underlying software/work; the upstream Hyper-YOLO citation appears in its README.

The inherited license is AGPL-3.0. Original copyright headers remain. See `THIRD_PARTY_NOTICES.md`. No package, dataset or checkpoint is relicensed by this release.
