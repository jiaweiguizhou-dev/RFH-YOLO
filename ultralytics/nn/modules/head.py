# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import math
from ultralytics.utils.torch_utils import make_divisible
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors

from .block import DFL, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_

__all__ = 'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder'

import math
import torch
import torch.nn as nn


import math

import torch
import torch.nn as nn







class Detect(nn.Module):
    """
    Dual-head detection module based on the Hyper-YOLO detection head.

    Input feature order:
        [A3, A4, A5, P3, P4, P5]

    Auxiliary features (A3-A5) provide training-only supervision.
    Main features (P3-P5) are used for training, validation, and inference.

    Training output:
        [aux_raw, main_raw]

    Validation and inference output:
        export=False: (decoded_main, main_raw)
        export=True: decoded_main

    The auxiliary detection head is excluded from inference and can
    be physically removed during deployment.
    """

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=(), inplace=True):
        """Initialize the auxiliary and main detection branches."""
        super().__init__()

        if len(ch) < 2 or len(ch) % 2 != 0:
            raise ValueError(
                "Detect requires an even number of feature channels: "
                "[auxiliary levels..., main levels...]."
            )

        self.nc = int(nc)
        self.nl = len(ch) // 2
        self.reg_max = 16
        self.no = self.nc + self.reg_max * 4
        self.inplace = inplace
        self.stride = torch.zeros(self.nl)

        self.dual = True
        self.main_only_deploy = False
        self.aux_loss_weight = 0.25

        aux_ch = ch[: self.nl]
        main_ch = ch[self.nl :]

        # Match the original Hyper-YOLO detection tower widths.
        c2 = max(aux_ch[0] // 4, self.reg_max * 4, 16)
        c3 = max(aux_ch[0], min(self.nc, 100))
        c4 = max(main_ch[0] // 4, self.reg_max * 4, 16)
        c5 = max(main_ch[0], min(self.nc, 100))

        # Auxiliary bounding-box regression towers.
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in aux_ch
        )

        # Auxiliary classification towers.
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in aux_ch
        )

        # Main bounding-box regression towers.
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c4, 3),
                Conv(c4, c4, 3),
                nn.Conv2d(c4, 4 * self.reg_max, 1),
            )
            for x in main_ch
        )

        # Main classification towers.
        self.cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c5, 3),
                Conv(c5, c5, 3),
                nn.Conv2d(c5, self.nc, 1),
            )
            for x in main_ch
        )

        self.dfl = (
            DFL(self.reg_max)
            if self.reg_max > 1
            else nn.Identity()
        )

    def switch_to_deploy(self):
        """
        Remove auxiliary detection towers for main-head-only inference.

        This conversion is irreversible for the current instance.
        The model must be in evaluation mode before conversion.
        Use DetectionModel.convert_to_deploy() to preserve the
        original training model.
        """
        if type(self) is not Detect:
            raise TypeError(
                "Auxiliary pruning supports the plain Detect head only."
            )

        if self.training:
            raise RuntimeError(
                "switch_to_deploy() requires model.eval()."
            )

        if getattr(self, "main_only_deploy", False):
            return self

        if (
            not isinstance(getattr(self, "f", None), (list, tuple))
            or len(self.f) != 2 * self.nl
        ):
            raise ValueError(
                "Detect.f must contain auxiliary sources "
                "followed by main sources."
            )

        if any(
            len(towers) != self.nl
            for towers in (
                self.cv2,
                self.cv3,
                self.cv4,
                self.cv5,
            )
        ):
            raise ValueError(
                "Detect tower counts do not match the dual-head layout."
            )

        if self.stride.numel() != self.nl:
            raise ValueError(
                "Detect.stride must already describe "
                "the main detection levels."
            )

        # Preserve the main feature sources.
        main_sources = list(self.f[self.nl :])

        # Physically remove auxiliary tower parameters.
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()

        self.f = main_sources
        self.dual = False
        self.main_only_deploy = True

        # Reset cached anchor information.
        self.shape = None
        self.anchors = self.stride.new_empty(0)
        self.strides = self.stride.new_empty(0)

        return self

    def forward(self, x):
        """
        Compute detection predictions.

        Both heads contribute predictions during training.
        Only the main head is evaluated during validation,
        inference, and export.
        """
        deploy = getattr(self, "main_only_deploy", False)

        if deploy and self.training:
            raise RuntimeError(
                "A pruned Detect is inference-only; "
                "use the original model for training."
            )

        expected = self.nl if deploy else self.nl * 2

        if not isinstance(x, (list, tuple)) or len(x) != expected:
            actual = (
                len(x)
                if isinstance(x, (list, tuple))
                else type(x)
            )
            raise ValueError(
                f"Detect expected {expected} feature maps, "
                f"got {actual}."
            )

        # Select feature maps according to the deployment state.
        if deploy:
            main_x = x
        else:
            aux_x = x[: self.nl]
            main_x = x[self.nl :]

        # Compute main-head predictions at all detection scales.
        main_raw = [
            torch.cat(
                (
                    self.cv4[i](main_x[i]),
                    self.cv5[i](main_x[i]),
                ),
                dim=1,
            )
            for i in range(self.nl)
        ]

        if self.training:
            # Return main predictions during stride initialization.
            if not bool(self.stride.detach().ne(0).any()):
                return main_raw

            # Compute auxiliary predictions during regular training.
            aux_raw = [
                torch.cat(
                    (
                        self.cv2[i](aux_x[i]),
                        self.cv3[i](aux_x[i]),
                    ),
                    dim=1,
                )
                for i in range(self.nl)
            ]

            return [aux_raw, main_raw]

        # Generate anchors when the feature-map shape changes.
        shape = main_x[0].shape

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                tensor.transpose(0, 1)
                for tensor in make_anchors(
                    main_raw,
                    self.stride,
                    0.5,
                )
            )
            self.shape = shape

        # Separate regression distributions and class predictions.
        box, cls = torch.cat(
            [
                pred.view(shape[0], self.no, -1)
                for pred in main_raw
            ],
            dim=2,
        ).split(
            (self.reg_max * 4, self.nc),
            dim=1,
        )

        # Decode bounding boxes using distribution focal loss.
        dbox = (
            dist2bbox(
                self.dfl(box),
                self.anchors.unsqueeze(0),
                xywh=True,
                dim=1,
            )
            * self.strides
        )

        # Combine decoded boxes and classification probabilities.
        y = torch.cat(
            (dbox, cls.sigmoid()),
            dim=1,
        )

        return y if self.export else (y, main_raw)

    def bias_init(self):
        """
        Initialize regression and classification prediction biases.

        Bias initialization is applied independently to the
        auxiliary and main detection towers using the
        corresponding detection strides.
        """
        # Initialize auxiliary detection tower biases.
        for reg, cls, stride in zip(
            self.cv2,
            self.cv3,
            self.stride,
        ):
            reg[-1].bias.data[:] = 1.0
            cls[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / stride) ** 2
            )

        # Initialize main detection tower biases.
        for reg, cls, stride in zip(
            self.cv4,
            self.cv5,
            self.stride,
        ):
            reg[-1].bias.data[:] = 1.0
            cls[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / stride) ** 2
            )


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
        padding, and groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # Training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = len(feats)
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
