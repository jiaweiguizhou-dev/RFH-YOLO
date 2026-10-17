# Third-party notices

This is a derivative research source tree, not an independently authored detector framework.
- Hyper-YOLO: https://github.com/iMoonLab/Hyper-YOLO — inherited README identifies the TPAMI 2025 work by Yifan Feng and coauthors; upstream repository identifies AGPL-3.0. Exact base commit was not present in this local non-Git checkout.
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics — local package version 8.0.227, inherited AGPL headers and root LICENSE retained byte-for-byte.
- Other vendored modules retain embedded source references and notices, including SAM, FastSAM, RT-DETR, tracking and export utilities. Their complete provenance has not been re-audited, following the user's instruction to skip further review. Do not interpret this notice as a completed license clearance.

Runtime dependencies remain external packages with their own license terms. requirements.txt includes torch, torchvision, numpy, OpenCV, Pillow, PyYAML, requests, scipy, matplotlib, pandas, seaborn, tqdm, tensorboard, psutil, py-cpuinfo, thop and pycocotools. Installation does not change their licensing.

Packaging changes dated 2026-10-17: portable wrapper and documentation added; original model Python/YAML files preserved; the inherited analytics endpoint and its embedded credential were removed in the release copy only, and event enablement requires a nonempty endpoint. This affects telemetry only. No training, loss, model graph or metric logic was changed.
