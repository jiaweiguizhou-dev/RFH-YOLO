import torch

import ultralytics.nn.tasks
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

torch.cuda.empty_cache()
print("✅ GPU 显存已清空")

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
  model = YOLO('ultralytics/cfg/models/hyper-yolo/hyper-yolon-repmanetpp.yaml')
  results = model.train(
    data=r"/root/sj-tmp/dataset.yaml",
    epochs=500,
    batch=20,
    imgsz=640,
    workers=8,
    device=0,  # Single-GPU training.
    optimizer='SGD',
    patience=100,
    save_period=50,

    amp=False,
    cache=False,  # Use disk caching only after measuring local SSD throughput.
    deterministic=True
  )
