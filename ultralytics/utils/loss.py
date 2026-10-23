# Ultralytics YOLO 🚀, AGPL-3.0 license
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.torch_utils import de_parallel
from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()



import os

import torch
import torch.nn as nn
import torch.nn.functional as F



def _hyp_get(hyp, name, default):
    """Read a hyperparameter from either a dict or a YOLOv8-style namespace."""
    if isinstance(hyp, dict):
        return hyp.get(name, default)
    return getattr(hyp, name, default)


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = pred.sigmoid()
        p_t = true * pred_prob + (1.0 - true) * (1.0 - pred_prob)
        alpha_factor = true * self.alpha + (1.0 - true) * (1.0 - self.alpha)
        loss *= alpha_factor * (1.0 - p_t).pow(self.gamma)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        bbox_mask = fg_mask.unsqueeze(-1).expand(-1, -1, 4)

        pred_bboxes_pos = torch.masked_select(
            pred_bboxes, bbox_mask
        ).view(-1, 4)
        target_bboxes_pos = torch.masked_select(
            target_bboxes, bbox_mask
        ).view(-1, 4)
        bbox_weight = torch.masked_select(
            target_scores.sum(-1), fg_mask
        ).unsqueeze(-1)

        iou = bbox_iou(
            pred_bboxes_pos,
            target_bboxes_pos,
            xywh=False,
            CIoU=True,
        )
        loss_iou = ((1.0 - iou) * bbox_weight).sum() / target_scores_sum

        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).expand(
                -1,
                -1,
                (self.reg_max + 1) * 4,
            )
            pred_dist_pos = torch.masked_select(
                pred_dist, dist_mask
            ).view(-1, 4, self.reg_max + 1)

            target_ltrb = bbox2dist(
                anchor_points,
                target_bboxes,
                self.reg_max,
            )
            target_ltrb_pos = torch.masked_select(
                target_ltrb, bbox_mask
            ).view(-1, 4)

            loss_dfl = self._df_loss(
                pred_dist_pos,
                target_ltrb_pos,
            ) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = pred_dist.new_zeros(())

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = 1.0 - weight_left

        loss_left = F.cross_entropy(
            pred_dist.reshape(-1, self.reg_max + 1),
            target_left.reshape(-1),
            reduction="none",
        ).view(target_left.shape) * weight_left

        loss_right = F.cross_entropy(
            pred_dist.reshape(-1, self.reg_max + 1),
            target_right.reshape(-1),
            reduction="none",
        ).view(target_left.shape) * weight_right

        return (loss_left + loss_right).mean(-1, keepdim=True)


import math

class v8DetectionLoss:
    """PD-DHAS: progress-dependent dual-head auxiliary supervision for detection."""

    def __init__(self, model):
        'init.'
        device = next(model.parameters()).device
        h = model.args

        
        m = model.model[-1]

        self.bce = nn.BCEWithLogitsLoss(
            reduction="none"
        )

        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = self.reg_max > 1

        # --------------------------------------------------
        
        # --------------------------------------------------
        
        #     aux_loss_weight=0.1
        #
        
        
        if isinstance(h, dict):
            self.aux_loss_weight = float(
                h.get(
                    "aux_loss_weight",
                    0.25
                )
            )
        else:
            self.aux_loss_weight = float(
                getattr(
                    h,
                    "aux_loss_weight",
                    0.25
                )
            )

        if self.aux_loss_weight < 0.0:
            raise ValueError(
                "aux_loss_weight must be greater "
                "than or equal to 0."
            )

        
        self.current_aux_loss_weight = self.aux_loss_weight

        # --------------------------------------------------
        
        # --------------------------------------------------
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0
        )

        
        self.bbox_loss = BboxLoss(
            self.reg_max - 1,
            use_dfl=self.use_dfl
        ).to(device)

        self.proj = torch.arange(
            self.reg_max,
            dtype=torch.float,
            device=device
        )

    # ======================================================
    
    # ======================================================
    def _get_hyp(self, name, default):
        'get hyp.'
        if isinstance(self.hyp, dict):
            return float(
                self.hyp.get(name, default)
            )

        return float(
            getattr(
                self.hyp,
                name,
                default
            )
        )

    # ======================================================
    
    # ======================================================
    @staticmethod
    def _scalar_value(value, default):
        'scalar value.'
        if value is None:
            return float(default)

        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(
                    "epoch and max_epochs must be scalar values."
                )
            value = value.detach().item()

        return float(value)

    # ======================================================
    
    # ======================================================
    def _get_aux_weight(self, batch, has_auxiliary):
        'get aux weight.'
        if (
            not has_auxiliary
            or self.aux_loss_weight == 0.0
        ):
            return 0.0

        epoch = self._scalar_value(
            batch.get(
                "epoch",
                0
            ),
            0
        )

        max_epochs = self._scalar_value(
            batch.get(
                "max_epochs",
                self._get_hyp(
                    "epochs",
                    1.0
                )
            ),
            1.0
        )

        
        #
        
        #   progress = epoch / (max_epochs - 1)
        #
        
        
        max_epochs = max(
            max_epochs,
            1.0
        )

        denominator = max(
            max_epochs - 1.0,
            1.0
        )

        progress = min(
            max(
                epoch / denominator,
                0.0
            ),
            1.0
        )

        warmup_end = 0.05
        decay_start = 0.50

        # ----------------------------------------------
        
        # ----------------------------------------------
        if progress < warmup_end:
            weight = (
                self.aux_loss_weight
                * progress
                / warmup_end
            )

        # ----------------------------------------------
        
        # ----------------------------------------------
        elif progress < decay_start:
            weight = self.aux_loss_weight

        # ----------------------------------------------
        
        # ----------------------------------------------
        else:
            decay_progress = (
                progress - decay_start
            ) / (
                1.0 - decay_start
            )

            weight = (
                self.aux_loss_weight
                * 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * decay_progress
                    )
                )
            )

        return float(
            max(
                weight,
                0.0
            )
        )

    # ======================================================
    
    # ======================================================
    def preprocess(
        self,
        targets,
        batch_size,
        scale_tensor
    ):
        'preprocess.'
        if targets.shape[0] == 0:
            out = torch.zeros(
                batch_size,
                0,
                5,
                device=self.device
            )

        else:
            image_indices = targets[:, 0]

            _, counts = image_indices.unique(
                return_counts=True
            )

            counts = counts.to(
                dtype=torch.int32
            )

            max_targets = int(
                counts.max().item()
            )

            out = torch.zeros(
                batch_size,
                max_targets,
                5,
                device=self.device,
                dtype=targets.dtype
            )

            for batch_index in range(batch_size):
                matches = (
                    image_indices == batch_index
                )

                num_matches = int(
                    matches.sum().item()
                )

                if num_matches > 0:
                    out[
                        batch_index,
                        :num_matches
                    ] = targets[
                        matches,
                        1:
                    ]

            
            out[..., 1:5] = xywh2xyxy(
                out[..., 1:5].mul_(
                    scale_tensor
                )
            )

        return out

    # ======================================================
    
    # ======================================================
    def bbox_decode(
        self,
        anchor_points,
        pred_dist
    ):
        'bbox decode.'
        if self.use_dfl:
            batch_size, num_anchors, channels = (
                pred_dist.shape
            )

            pred_dist = pred_dist.view(
                batch_size,
                num_anchors,
                4,
                channels // 4
            )

            pred_dist = pred_dist.softmax(
                dim=3
            ).matmul(
                self.proj.type(
                    pred_dist.dtype
                )
            )

        return dist2bbox(
            pred_dist,
            anchor_points,
            xywh=False
        )

    # ======================================================
    
    # ======================================================
    @staticmethod
    def _is_dual_predictions(preds):
        'is dual predictions.'
        return (
            isinstance(
                preds,
                (list, tuple)
            )
            and len(preds) == 2
            and isinstance(
                preds[0],
                (list, tuple)
            )
            and isinstance(
                preds[1],
                (list, tuple)
            )
        )

    # ======================================================
    
    # ======================================================
    def _split_predictions(self, preds):
        'split predictions.'
        
        #     (decoded_predictions, main_feats)
        #
        
        if isinstance(preds, tuple):
            raw_preds = preds[1]
        else:
            raw_preds = preds

        
        if self._is_dual_predictions(
            raw_preds
        ):
            aux_feats = raw_preds[0]
            main_feats = raw_preds[1]

            return aux_feats, main_feats

        
        return None, raw_preds

    # ======================================================
    
    # ======================================================
    def _check_feature_shapes(
        self,
        aux_feats,
        main_feats
    ):
        'check feature shapes.'
        if not isinstance(
            main_feats,
            (list, tuple)
        ):
            raise TypeError(
                "main_feats must be a list or tuple "
                "of prediction feature maps."
            )

        if len(main_feats) != len(
            self.stride
        ):
            raise RuntimeError(
                "The number of main prediction levels "
                f"is {len(main_feats)}, but Detect.stride "
                f"contains {len(self.stride)} levels."
            )

        if aux_feats is None:
            return

        if len(aux_feats) != len(
            main_feats
        ):
            raise RuntimeError(
                "Auxiliary and main heads must have "
                "the same number of prediction levels. "
                f"Got auxiliary={len(aux_feats)}, "
                f"main={len(main_feats)}."
            )

        for level, (
            aux_feat,
            main_feat
        ) in enumerate(
            zip(aux_feats, main_feats)
        ):
            if aux_feat.shape[0] != main_feat.shape[0]:
                raise RuntimeError(
                    f"Batch size mismatch at level {level}: "
                    f"auxiliary={aux_feat.shape[0]}, "
                    f"main={main_feat.shape[0]}."
                )

            if (
                aux_feat.shape[-2:]
                != main_feat.shape[-2:]
            ):
                raise RuntimeError(
                    "Auxiliary and main feature maps "
                    "must have equal spatial sizes. "
                    f"Level {level}: "
                    f"auxiliary={aux_feat.shape[-2:]}, "
                    f"main={main_feat.shape[-2:]}."
                )

            if aux_feat.shape[1] != self.no:
                raise RuntimeError(
                    f"Auxiliary output channels at level "
                    f"{level} must equal no={self.no}, "
                    f"but got {aux_feat.shape[1]}."
                )

            if main_feat.shape[1] != self.no:
                raise RuntimeError(
                    f"Main output channels at level "
                    f"{level} must equal no={self.no}, "
                    f"but got {main_feat.shape[1]}."
                )

    # ======================================================
    
    # ======================================================
    def _flatten_predictions(
        self,
        feats
    ):
        'flatten predictions.'
        batch_size = feats[0].shape[0]

        predictions = torch.cat(
            [
                feat.view(
                    batch_size,
                    self.no,
                    -1
                )
                for feat in feats
            ],
            dim=2
        )

        pred_distri, pred_scores = (
            predictions.split(
                (
                    self.reg_max * 4,
                    self.nc
                ),
                dim=1
            )
        )

        pred_scores = pred_scores.permute(
            0,
            2,
            1
        ).contiguous()

        pred_distri = pred_distri.permute(
            0,
            2,
            1
        ).contiguous()

        return pred_distri, pred_scores

    # ======================================================
    
    # ======================================================
    def __call__(self, preds, batch):
        'call.'
        
        # [box, cls, dfl]
        main_loss = torch.zeros(
            3,
            device=self.device
        )

        aux_loss = torch.zeros(
            3,
            device=self.device
        )

        # --------------------------------------------------
        
        # --------------------------------------------------
        aux_feats, main_feats = (
            self._split_predictions(preds)
        )

        self._check_feature_shapes(
            aux_feats,
            main_feats
        )

        has_auxiliary = aux_feats is not None

        # --------------------------------------------------
        
        # --------------------------------------------------
        (
            main_pred_distri,
            main_pred_scores
        ) = self._flatten_predictions(
            main_feats
        )

        if has_auxiliary:
            (
                aux_pred_distri,
                aux_pred_scores
            ) = self._flatten_predictions(
                aux_feats
            )
        else:
            aux_pred_distri = None
            aux_pred_scores = None

        dtype = main_pred_scores.dtype
        batch_size = main_pred_scores.shape[0]

        # --------------------------------------------------
        
        # --------------------------------------------------
        imgsz = torch.tensor(
            main_feats[0].shape[2:],
            device=self.device,
            dtype=dtype
        ) * self.stride[0]

        anchor_points, stride_tensor = (
            make_anchors(
                main_feats,
                self.stride,
                0.5
            )
        )

        # --------------------------------------------------
        
        # --------------------------------------------------
        targets = torch.cat(
            (
                batch["batch_idx"].view(
                    -1,
                    1
                ),
                batch["cls"].view(
                    -1,
                    1
                ),
                batch["bboxes"]
            ),
            dim=1
        )

        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=imgsz[
                [1, 0, 1, 0]
            ]
        )

        gt_labels, gt_bboxes = (
            targets.split(
                (1, 4),
                dim=2
            )
        )

        mask_gt = gt_bboxes.sum(
            dim=2,
            keepdim=True
        ).gt_(0)

        # --------------------------------------------------
        
        # --------------------------------------------------
        main_pred_bboxes = self.bbox_decode(
            anchor_points,
            main_pred_distri
        )

        if has_auxiliary:
            aux_pred_bboxes = self.bbox_decode(
                anchor_points,
                aux_pred_distri
            )
        else:
            aux_pred_bboxes = None

        # --------------------------------------------------
        
        # --------------------------------------------------
        
        
        #
        
        
        
        assign_results = self.assigner(
            main_pred_scores.detach().sigmoid(),

            (
                main_pred_bboxes.detach()
                * stride_tensor
            ).type(
                gt_bboxes.dtype
            ),

            anchor_points * stride_tensor,

            gt_labels,
            gt_bboxes,
            mask_gt
        )

        if len(assign_results) == 5:
            (
                _,
                target_bboxes,
                target_scores,
                fg_mask,
                _
            ) = assign_results

        elif len(assign_results) == 4:
            (
                _,
                target_bboxes,
                target_scores,
                fg_mask
            ) = assign_results

        else:
            raise RuntimeError(
                "Unexpected TaskAlignedAssigner output "
                f"length: {len(assign_results)}."
            )

        target_scores_sum = (
            target_scores.sum()
            .clamp_min(1.0)
        )

        # --------------------------------------------------
        
        # --------------------------------------------------
        main_loss[1] = self.bce(
            main_pred_scores,
            target_scores.to(
                main_pred_scores.dtype
            )
        ).sum() / target_scores_sum

        if has_auxiliary:
            aux_loss[1] = self.bce(
                aux_pred_scores,
                target_scores.to(
                    aux_pred_scores.dtype
                )
            ).sum() / target_scores_sum

        # --------------------------------------------------
        
        # --------------------------------------------------
        if fg_mask.any():
            target_bboxes_grid = (
                target_bboxes
                / stride_tensor
            )

            (
                main_loss[0],
                main_loss[2]
            ) = self.bbox_loss(
                main_pred_distri,
                main_pred_bboxes,
                anchor_points,
                target_bboxes_grid,
                target_scores,
                target_scores_sum,
                fg_mask
            )

            if has_auxiliary:
                (
                    aux_loss[0],
                    aux_loss[2]
                ) = self.bbox_loss(
                    aux_pred_distri,
                    aux_pred_bboxes,
                    anchor_points,
                    target_bboxes_grid,
                    target_scores,
                    target_scores_sum,
                    fg_mask
                )

        # --------------------------------------------------
        
        # --------------------------------------------------
        gains = main_loss.new_tensor(
            [
                self._get_hyp(
                    "box",
                    7.5
                ),
                self._get_hyp(
                    "cls",
                    0.5
                ),
                self._get_hyp(
                    "dfl",
                    1.5
                )
            ]
        )

        main_loss *= gains
        aux_loss *= gains

        # --------------------------------------------------
        
        # --------------------------------------------------
        aux_weight = self._get_aux_weight(
            batch,
            has_auxiliary
        )

        self.current_aux_loss_weight = aux_weight

        
        #
        #   loss[0] =
        #       g_box * (
        #           L_box_main
        #           + lambda_aux(e) * L_box_aux
        #       )
        #
        #   loss[1] =
        #       g_cls * (
        #           L_cls_main
        #           + lambda_aux(e) * L_cls_aux
        #       )
        #
        #   loss[2] =
        #       g_dfl * (
        #           L_dfl_main
        #           + lambda_aux(e) * L_dfl_aux
        #       )
        loss = (
            main_loss
            + aux_weight * aux_loss
        )

        # --------------------------------------------------
        
        # --------------------------------------------------
        return (
            loss.sum() * batch_size,
            loss.detach()
        )



class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # Masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            loss[1] = self.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto,
                                                       pred_masks, imgsz, self.overlap)

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor,
                         area: torch.Tensor) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i],
                                              marea_i[fg_mask_i])

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx,
                                                             stride_tensor, target_bboxes, pred_kpts)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes,
                                 pred_kpts):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros((batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
                                        device=keypoints.device)

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, :keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2]))

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='mean')
        loss_items = loss.detach()
        return loss, loss_items


