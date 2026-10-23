# RFH-YOLO

**RFH-YOLO: Relational-Field Hypergraph Modeling for Batik Pattern Detection**

This repository provides the source code for RFH-YOLO.

RFH-YOLO is developed based on [Hyper-YOLO](https://github.com/iMoonLab/Hyper-YOLO) and a modified Ultralytics YOLO 8.0.227 source tree.

The repository includes the model implementation, training and evaluation scripts, model configurations, experimental records, and trained model checkpoints.

Publication, checkpoint distribution, and dataset access are subject to author approval and applicable permissions.

## Layout

- `train.py`: main single-GPU training entry point.
- `test.py`: main test-set evaluation and image prediction entry point, including auxiliary-head pruning and convolution fusion.
- `train_repmanetpp_original.py`, `test_repmanetpp_original.py`: alternative training and evaluation scripts using different model configurations.
- `ultralytics/`: modified local implementation, including `nn/modules/block.py` (RFHM, RepMANet, RepMixConv), `nn/modules/head.py` (dual-head Detect), `utils/loss.py` (progress-dependent auxiliary supervision), and `nn/tasks.py` (`convert_to_deploy`).
- `ultralytics/cfg/models/hyper-yolo/hyper-yolon-repmanet.yaml`: main model configuration file.
- `runs/train2/args.yaml`, `runs/train2/results.csv`: recorded training arguments and training logs.
- `runs/train2/weights/best.pt`, `runs/train2/weights/epoch300.pt`: trained model checkpoints.
- `environment.yaml`, `requirements.txt`: environment and dependency specifications.

## Environment and installation

The `environment.yaml` includes Python 3.8.16, PyTorch 2.0.1, torchvision 0.15.2, and CUDA 11.7-related packages.

The current `train.py` uses `torch.serialization.add_safe_globals`, which requires a newer PyTorch version than 2.0.1.

Install a compatible PyTorch, torchvision, and CUDA combination according to your NVIDIA driver and system configuration. See the [official PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

```bash
conda create -n rfh-yolo python=3.10 -y
conda activate rfh-yolo

# Install compatible PyTorch and torchvision versions for your CUDA environment and NVIDIA driver.

python -m pip install "numpy<2" -r requirements.txt
python -m pip install -e . --no-deps

python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__file__); print(hasattr(torch.serialization, 'add_safe_globals'))"
```

Run all commands from the repository root. Confirm that `ultralytics.__file__` points to the local RFH-YOLO implementation rather than another Ultralytics installation from PyPI.

For reproducibility, record the Python package versions (`python -m pip freeze`), GPU model, NVIDIA driver version, and CUDA environment.

**Security note:** Load only model checkpoints from trusted sources. The checkpoint loaders use Python pickle deserialization, and `test.py` may load checkpoints with `weights_only=False`.

## BatikDet

**BatikDet images, annotations, and the external dataset YAML file are not included in this repository.**

The main scripts reference `/root/sj-tmp/dataset.yaml`; `test.py` also references `/root/sj-tmp/test/images`. Modify these paths to point to your local dataset before running the scripts.

When reproducing the experimental results, preserve the original training, validation, and test splits, as well as the corresponding class IDs.

```text
BatikDet/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
```

YOLO annotation files follow the `class_id x_center y_center width height` format, using normalized coordinates and zero-based class IDs from 0 to 15. Each image should have a corresponding `.txt` annotation file with the same filename stem.

The BatikDet dataset described in the manuscript contains **9,777 images and 75,249 annotated instances**, with **7,820/978/979 images** in the training, validation, and test splits, respectively.

The 16 pattern categories, in class ID order, are: Bird, Drum, Butterfly, Dragon, Fish, Human, Pomegranate, Vortex, Pomegranate Flower, Lotus, Interlocking Lozenge, Cloud, Meander, Bat, Ox, and Longevity.

### Dataset Availability

The BatikDet dataset contains 9,777 images and 75,249 annotated instances across 16 batik pattern categories.

The dataset is available upon reasonable request for academic research purposes.

To request access, please contact the corresponding author at **[wjia@gzu.edu.cn](mailto:wjia@gzu.edu.cn)**.

Dataset access is subject to applicable data-use agreements and copyright restrictions.

## Train

Modify the `data` path in `train.py` and run:

```bash
python train.py
```

The main training script loads `ultralytics/cfg/models/hyper-yolo/hyper-yolon-repmanet.yaml`.

The training configuration includes **500 planned epochs, batch size 20, image size 640, the SGD optimizer, GPU 0, 8 data-loading workers, early-stopping patience of 100, deterministic mode enabled, and AMP disabled**.

The saved `runs/train2/args.yaml` additionally records `pretrained: false`, `seed: 0`, `lr0: 0.01`, `momentum: 0.937`, `weight_decay: 0.0005`, and box/classification/DFL loss weights of **7.5/0.5/1.5**.

Early stopping may terminate training before reaching 500 epochs. The `nc: 80` setting in the model YAML is a template value and is overridden by the 16-class dataset YAML during training.

The RFH-YOLO architecture consists of three main components:

- **RFHM (Relational-Field Hypergraph Module):** Models higher-order relationships among spatial features through global and local relation scales, adaptive hypergraph membership, and learnable residual gating.
- **RepMANet (Re-parameterized Multi-branch Aggregation Network):** Enhances feature aggregation using re-parameterizable convolutional branches. RepMixConv supports structural conversion into a single `3×3` convolution during deployment.
- **PD-DHAS (Progress-Dependent Dual-Head Auxiliary Supervision):** Introduces a progress-dependent auxiliary loss with a maximum weight of **0.25**. The auxiliary weight is warmed up during the first 5% of training, remains constant until 50% of training progress, and follows cosine decay during the remaining training process.

`Detect` receives six feature maps, `[B3, B4, B5, N3, N4, N5]`, where the first three are auxiliary detection features and the last three are main detection features. Only the main detection branch is used during inference.

The current dual-head implementation requires six input feature maps. Setting the auxiliary loss weight to zero does not remove auxiliary-head computation. Therefore, detection YAML configurations designed for three input features should not be used directly with the current six-input `Detect` module.

## Validate and predict

The main `test.py` script uses the following configuration variables. Modify them according to your local paths before running:

```python
WEIGHTS = "runs/train2/weights/best.pt"
DATA_YAML = "/root/sj-tmp/dataset.yaml"
VAL_IMAGES = "/root/sj-tmp/test/images"
SAVE_ROOT = "runs/RFH_YOLO_Aug/DA5_seed02/weights/"
```

```bash
python test.py
```

The script first loads the model checkpoint, uses `convert_to_deploy(fuse=False)` to remove auxiliary detection branches, invokes `model.fuse()` for convolution fusion, checks the RepMixConv fusion process, evaluates the model on the **test split**, and performs image prediction on `VAL_IMAGES`.

Validation uses `imgsz=640`, `batch=20`, and `device=0`.

Prediction uses `conf=0.25` and `iou=0.7`, and saves detection results, text labels, confidence scores, and cropped detection regions.

The validation call does not explicitly specify `conf`, NMS IoU, or `max_det`; the saved training record `args.yaml` contains `conf=0.001`, `iou=0.7`, and `max_det=300`.

`SAVE_ROOT` controls logging and validation output but is not explicitly passed as the prediction `project` parameter. Prediction results are saved according to the configuration used by the prediction interface.

Validation timing differs from the synchronized batch-1 FPS benchmarking protocol used for inference-speed measurement.

## Deploy

`DetectionModel.convert_to_deploy(fuse=True)` returns an evaluation-mode model copy with the auxiliary detection branches removed and supported re-parameterizable convolutional branches fused.

```python
# Example after loading a trusted model checkpoint with ultralytics.YOLO:
deploy_model = yolo_model.model.convert_to_deploy(fuse=True)
```

The deployment conversion performs the following operations:

- Removes the auxiliary detection branches.
- Retains the main detection branch for inference.
- Fuses supported re-parameterizable convolutional branches.
- Simplifies the model structure for deployment.

When deploying the model on a target platform, verify numerical equivalence before and after model conversion, checkpoint loading, and inference performance.

The current deployment workflow does not include ONNX/TensorRT export functionality.

## Citation and license

If you use RFH-YOLO in your research, please cite the corresponding publication:

**RFH-YOLO: Relational-Field Hypergraph Modeling for Batik Pattern Detection**

The complete bibliographic information will be provided once the final publication details are confirmed.

Please also acknowledge the following foundational research and software projects:

- [Hyper-YOLO](https://github.com/iMoonLab/Hyper-YOLO)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

The original Hyper-YOLO citation information is available in the README of its upstream GitHub repository.

This repository contains code modified from Hyper-YOLO and Ultralytics YOLO. The inherited open-source license is **AGPL-3.0**, and applicable upstream copyright notices and license requirements must be retained.
