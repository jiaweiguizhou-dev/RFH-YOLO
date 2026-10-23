from pathlib import Path
import inspect
import sys
import logging
import torch


# ============================================================


# ============================================================
_real_torch_load = torch.load

def torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)

torch.load = torch_load_compat


# ==============================

# ==============================
WEIGHTS = "runs/train2/weights/best.pt"
DATA_YAML = "/root/sj-tmp/dataset.yaml"
VAL_IMAGES = "/root/sj-tmp/test/images"
SAVE_ROOT = "runs/RFH_YOLO_Aug/DA5_seed02/weights/"

Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)

LOG_TXT = Path(SAVE_ROOT) / "yoloours_test_report.txt"


# ==============================

# ==============================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def add_ultralytics_file_logger(log_path):
    'add ultralytics file logger.'
    try:
        from ultralytics.utils import LOGGER

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)

        LOGGER.addHandler(file_handler)

    except Exception as e:
        print(f"添加 Ultralytics 文件日志失败：{e}")


def print_runtime_environment():
    'print runtime environment.'
    import ultralytics
    import ultralytics.nn.modules.block as block_module

    actual_package = Path(ultralytics.__file__).resolve()
    actual_block = Path(block_module.__file__).resolve()
    expected_block = (
        Path(__file__).resolve().parent
        / "ultralytics"
        / "nn"
        / "modules"
        / "block.py"
    ).resolve()

    print("\n========== 运行时源码与设备 ==========")
    print(f"Ultralytics package: {actual_package}")
    print(f"Loaded block.py:     {actual_block}")
    print(f"Expected block.py:   {expected_block}")
    print(f"Source path match:   {actual_block == expected_block}")
    print(f"RepMixConv source:   {Path(inspect.getsourcefile(block_module.RepMixConv)).resolve()}")
    print(f"RepMixBottleneck:    {Path(inspect.getsourcefile(block_module.RepMixBottleneck)).resolve()}")
    print(f"RepMANet source:   {Path(inspect.getsourcefile(block_module.RepMANet)).resolve()}")
    print(f"PyTorch:             {torch.__version__}")
    print(f"CUDA available:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:         {torch.cuda.get_device_name(0)}")

    if actual_block != expected_block:
        print(
            "WARNING: 当前运行时没有加载本工程中的 block.py，"
            "后续融合结果不能代表本次修改。"
        )


def audit_repmix_fusion(model, stage):
    'audit repmix fusion.'
    core_model = getattr(model, "model", model)
    modules = [
        module
        for module in core_model.modules()
        if module.__class__.__name__ == "RepMixConv"
    ]
    training_branch_names = (
        "rbr_3x3",
        "rbr_1x1",
    )
    legacy_branch_names = (
        "dwconv_w",
        "dwconv_h",
        "rbr_pw",
        "rbr_spatial_bn",
        "gc",
        "identity_channels",
        "split_indexes",
    )

    print(f"\n========== RepMixConv 融合审计：{stage} ==========")
    print(f"RepMixConv count: {len(modules)}")

    if not modules:
        print("WARNING: 模型中没有找到 RepMixConv，请检查权重和源码版本。")
        return False

    fully_fused_count = 0
    for index, module in enumerate(modules):
        has_reparam = hasattr(module, "rbr_reparam")
        remaining_branches = [
            name for name in training_branch_names if hasattr(module, name)
        ]
        legacy_branches = [
            name for name in legacy_branch_names if hasattr(module, name)
        ]
        reparam = getattr(module, "rbr_reparam", None)
        is_single_full_3x3 = (
            isinstance(reparam, torch.nn.Conv2d)
            and reparam.kernel_size == (3, 3)
            and reparam.stride == (module.s, module.s)
            and reparam.padding == (1, 1)
            and reparam.groups == module.g
            and reparam.bias is not None
        )
        is_fully_fused = (
            has_reparam
            and not remaining_branches
            and not legacy_branches
            and is_single_full_3x3
        )
        fully_fused_count += int(is_fully_fused)

        print(
            f"[{index:02d}] "
            f"fully_fused={is_fully_fused}, "
            f"rbr_reparam={has_reparam}, "
            f"single_full_3x3={is_single_full_3x3}, "
            f"remaining={remaining_branches}, "
            f"legacy={legacy_branches}, "
            f"PATH={'single equivalent 3x3' if is_fully_fused else '3x3 + 1x1'}, "
            f"FUSED={is_fully_fused}"
        )

    all_fused = fully_fused_count == len(modules)
    print(
        f"Fusion summary: {fully_fused_count}/{len(modules)} "
        f"RepMixConv modules are single Full 3x3 Conv"
    )
    print(f"All RepMixConv fused: {all_fused}")
    return all_fused


def get_model_complexity(model):
    'get model complexity.'
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    return total_params, trainable_params


def summarize_metrics(model, metrics):
    'summarize metrics.'
    total_params, trainable_params = get_model_complexity(model)

    params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6

    speed = getattr(metrics, "speed", None)

    if speed is not None:
        preprocess = speed.get("preprocess", 0.0)
        inference = speed.get("inference", 0.0)
        loss = speed.get("loss", 0.0)
        postprocess = speed.get("postprocess", 0.0)

        total_time = preprocess + inference + loss + postprocess
        inference_fps = 1000.0 / inference if inference > 0 else 0.0
        fps = 1000.0 / total_time if total_time > 0 else 0.0
    else:
        preprocess = inference = loss = postprocess = 0.0
        total_time = inference_fps = fps = 0.0

    print("\n========== 模型性能汇总 ==========")
    print(f"Params(M): {params_m:.2f}")
    print(f"Trainable Params(M): {trainable_params_m:.2f}")
    print(f"Preprocess(ms): {preprocess:.2f}")
    print(f"Inference(ms): {inference:.2f}")
    print(f"Loss(ms): {loss:.2f}")
    print(f"Postprocess(ms): {postprocess:.2f}")
    print(f"单图总耗时(ms): {total_time:.2f}")
    print(f"纯模型 Inference FPS: {inference_fps:.2f}")
    print(f"端到端 FPS: {fps:.2f}")
    print(
        "说明：纯模型 FPS 只使用 metrics.speed['inference']；"
        "端到端 FPS 还包含 preprocess、loss 计时和 NMS postprocess。"
    )


def validate_model(model):
    model.model.eval()

    print("\n========== 开始验证集测试 ==========")
    print(f"Weights: {WEIGHTS}")
    print(f"Data: {DATA_YAML}")
    print(f"Save root: {SAVE_ROOT}")

    print("\n========== 模型结构信息 ==========")
    model.info(verbose=True)
    audit_repmix_fusion(model, "model.val() 之前")

    with torch.no_grad():
        metrics = model.val(
            data=DATA_YAML,
            split="test",
            imgsz=640,
            batch=20,
            device=0,
            save=True,
            save_txt=True,
            save_conf=True,
            project=SAVE_ROOT,
            name="val_metrics",
            exist_ok=True,
            verbose=True
        )

    audit_repmix_fusion(model, "model.val() 之后")
    summarize_metrics(model, metrics)

    return metrics


def predict_val_images(model):
    model.model.eval()
    audit_repmix_fusion(model, "model.predict() 之前")

    print("\n========== 开始保存验证集检测结果 ==========")
    print(f"Source: {VAL_IMAGES}")

    with torch.no_grad():
        model.predict(
            source=VAL_IMAGES,
            imgsz=640,
            conf=0.25,
            iou=0.7,
            device=0,
            save=True,
            save_txt=True,
            save_conf=True,
            save_crop=True,

            name="val_predictions",
            exist_ok=True,
            verbose=False
        )

    print("\n========== 检测结果保存完成 ==========")
    print(f"Prediction results saved to: {Path(SAVE_ROOT) / 'val_predictions'}")


def main():
    
    from ultralytics import YOLO

    add_ultralytics_file_logger(LOG_TXT)
    print_runtime_environment()

    model = YOLO(WEIGHTS)
    model.model.eval()
    # Explicit deployment conversion: replace the wrapper's core with the
    # pruned copy so validation and get_model_complexity use that same copy.
    # Fusion alone does not remove Detect's auxiliary prediction towers.
    model.model = model.model.convert_to_deploy(fuse=False)
    head = model.model.model[-1]
    if (
        not getattr(head, "main_only_deploy", False)
        or len(head.f) != head.nl
        or len(head.cv2) != 0
        or len(head.cv3) != 0
    ):
        raise RuntimeError("Auxiliary head pruning did not produce a main-only Detect.")
    print("\n========== Detect 部署裁剪状态 ==========")
    print(f"Main-only deployment: {head.main_only_deploy}")
    print(f"Detect input sources: {head.f}")
    print(f"Auxiliary regression/classification towers: {len(head.cv2)}/{len(head.cv3)}")
    print(f"Main regression/classification towers: {len(head.cv4)}/{len(head.cv5)}")
    audit_repmix_fusion(model, "独立模型融合之前")
    model.fuse()
    if not audit_repmix_fusion(model, "独立模型融合之后"):
        raise RuntimeError("RepMixConv fusion did not produce the expected deployment graph.")

    validate_model(model)
    predict_val_images(model)


if __name__ == "__main__":
    with open(LOG_TXT, "w", encoding="utf-8") as f:
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        sys.stdout = Tee(sys.stdout, f)
        sys.stderr = Tee(sys.stderr, f)

        try:
            main()
        except FileNotFoundError as e:
            print(f"文件路径不存在：{e}")
        except RuntimeError as e:
            print(f"GPU运行错误：{e}")
        except Exception as e:
            print(f"程序异常：{e}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    print(f"\n日志已保存到: {LOG_TXT}")
