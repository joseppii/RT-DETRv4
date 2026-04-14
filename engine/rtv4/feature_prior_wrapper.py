"""ONNX export wrapper for RT-DETRv4 with spatial + content feature priors.

Extends the spatial prior path with a cached-feature "content" prior injected
at decoder slot 299. Also exposes encoder tokens as a named output so the C++
adapter can extract per-detection features for post-filter ranking.
"""

import torch
import torch.nn as nn


class FeaturePriorExportWrapper(nn.Module):
    """Wraps a deployed RTv4 model for ONNX export with spatial + content priors.

    Inputs:
        images              [1, 3, 640, 640]  float32
        spatial_prior       [1, 4]            float32  — [cx, cy, w, h] normalised [0,1]
        use_spatial_prior   [1]               int64    — 1 = inject, 0 = normal
        content_prior       [1, 256]          float32  — L2-normalised encoder token
        use_content_prior   [1]               int64    — 1 = inject, 0 = normal

    Outputs:
        detections          [1, 300, 4 + num_classes]  float32
            [:, :, 0:4]  = pred_boxes (normalised cxcywh in [0,1])
            [:, :, 4:]   = sigmoid(pred_logits)
        encoder_features    [1, N_tokens, 256]  float32
            Raw encoder memory tokens across FPN levels (concatenated in the
            order the encoder outputs them). N_tokens = sum of H*W per level
            for the fixed 640x640 input (typically 8400 for strides 8/16/32).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, spatial_prior, use_spatial_prior,
                content_prior, use_content_prior):
        outputs = self.model(
            images,
            spatial_prior=spatial_prior,
            use_spatial_prior=use_spatial_prior,
            content_prior=content_prior,
            use_content_prior=use_content_prior,
        )
        boxes = outputs["pred_boxes"]                # [B, 300, 4]
        scores = torch.sigmoid(outputs["pred_logits"])  # [B, 300, num_classes]
        detections = torch.cat([boxes, scores], dim=-1)  # [B, 300, 4+num_classes]
        encoder_features = outputs["encoder_features"]   # [B, N_tokens, 256]
        return detections, encoder_features
