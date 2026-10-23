import argparse
import inspect
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "train2" / "weights" / "best.pt"
DEFAULT_DATA = Path("/root/sj-tmp/dataset.yaml")
TRAINING_BRANCHES = (
    "rbr_3x3",
    "rbr_1x1",
)
LEGACY_BRANCHES = (
    "dwconv_w",
    "dwconv_h",
    "rbr_pw",
    "rbr_spatial_bn",
    "rbr_spatial_reparam",
    "gc",
    "identity_channels",
    "split_indexes",
)


def install_trusted_checkpoint_compatibility():
    """Allow weights_only=False only after the caller explicitly marks the checkpoint trusted."""
    real_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return real_torch_load(*args, **kwargs)

    torch.load = trusted_torch_load
    print("WARNING: trusted checkpoint mode enabled; torch.load(weights_only=False) may execute pickle code.")


def print_runtime_environment():
    import ultralytics
    import ultralytics.nn.modules.block as block_module

    expected_block = PROJECT_ROOT / "ultralytics" / "nn" / "modules" / "block.py"
    actual_block = Path(block_module.__file__).resolve()
    print(f"Ultralytics package: {Path(ultralytics.__file__).resolve()}")
    print(f"Loaded block.py:     {actual_block}")
    print(f"Expected block.py:   {expected_block.resolve()}")
    print(f"RepMixConv source:   {Path(inspect.getsourcefile(block_module.RepMixConv)).resolve()}")
    print(f"RepMixBottleneck:    {Path(inspect.getsourcefile(block_module.RepMixBottleneck)).resolve()}")
    print(f"RepMANet source:   {Path(inspect.getsourcefile(block_module.RepMANet)).resolve()}")
    if actual_block != expected_block.resolve():
        raise RuntimeError("Loaded Ultralytics is not the source tree beside this test script.")


def audit_checkpoint_structure(model):
    core_model = getattr(model, "model", model)
    modules = [m for m in core_model.modules() if m.__class__.__name__ == "RepMixConv"]
    print(f"RepMixConv count: {len(modules)}")
    if not modules:
        raise RuntimeError("Checkpoint contains no RepMixConv modules.")

    for index, module in enumerate(modules):
        legacy = [name for name in LEGACY_BRANCHES if hasattr(module, name)]
        if legacy:
            raise RuntimeError(
                f"RepMixConv[{index}] uses incompatible legacy branches {legacy}; "
                "this checkpoint was trained with the removed directional RepMixConv."
            )

        present_training = [name for name in TRAINING_BRANCHES if hasattr(module, name)]
        reparam = getattr(module, "rbr_reparam", None)
        fused = (
            isinstance(reparam, torch.nn.Conv2d)
            and reparam.kernel_size == (3, 3)
            and reparam.stride == (module.s, module.s)
            and reparam.padding == (1, 1)
            and reparam.groups == module.g
            and reparam.bias is not None
            and not present_training
        )
        training = len(present_training) == len(TRAINING_BRANCHES) and reparam is None
        if not (training or fused):
            raise RuntimeError(
                f"RepMixConv[{index}] has an incomplete/incompatible structure: "
                f"training branches={present_training}, fused={fused}."
            )

    return modules


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a locked RepMANet-original checkpoint.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = args.weights.resolve()
    data = args.data.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if args.trusted_checkpoint:
        install_trusted_checkpoint_compatibility()

    print_runtime_environment()
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    print(model.model)
    audit_checkpoint_structure(model)
    model.model.eval()
    model.fuse()
    audit_checkpoint_structure(model)
    model.val(
        data=str(data),
        split="test",
        imgsz=640,
        batch=20,
        device=0,
    )


if __name__ == "__main__":
    main()
