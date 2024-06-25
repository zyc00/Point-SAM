"""Segment Anything Model for Point Clouds.

References:
- https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/sam.py
"""

from typing import Dict, List

import torch
import torch.nn as nn

from .common import repeat_interleave
from .mask_decoder import AuxInputs, MaskDecoder
from .pc_encoder import PointCloudEncoder
from .prompt_encoder import MaskEncoder, PointEncoder


class PointCloudSAM(nn.Module):
    def __init__(
        self,
        pc_encoder: PointCloudEncoder,
        mask_encoder: MaskEncoder,
        mask_decoder: MaskDecoder,
    ):
        super().__init__()
        self.pc_encoder = pc_encoder
        self.point_encoder = PointEncoder(pc_encoder.embed_dim)
        self.mask_encoder = mask_encoder
        self.mask_decoder = mask_decoder

    def set_pointcloud(self, coords: torch.Tensor, features: torch.Tensor):
        self.pc_embeddings, self.patches = self.pc_encoder(coords, features)
        self.coords = coords
        self.features = features

    def predict_masks(
        self,
        prompt_coords: torch.Tensor,
        prompt_labels: torch.Tensor,
        prompt_masks: torch.Tensor = None,
        multimask_output: bool = True,
    ):
        """Predict masks given point prompts.

        Args:
            coords: [B, N, 3]. Point cloud coordinates, normalized to [-1, 1].
            features: [B, N, F]. Point cloud features.
        """
        # pc_embeddings: [B, num_patches, D]
        pc_embeddings, patches = self.pc_embeddings, self.patches
        centers = patches["centers"]  # [B, num_patches, 3]
        knn_idx = patches["knn_idx"]  # [B, N, K]
        aux_inputs = AuxInputs(
            coords=self.coords, features=self.features, centers=centers
        )

        # [B, num_patches, D]
        pc_pe = self.point_encoder.pe_layer(centers)

        # [B * M, num_queries, D]
        sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)

        # [B * M, num_patches, D] or [B, num_patches, D] (if prompt_masks=None)
        dense_embeddings = self.mask_encoder(
            prompt_masks, self.coords, centers, knn_idx
        )

        # [B * M, num_patches, D]
        dense_embeddings = repeat_interleave(
            dense_embeddings,
            sparse_embeddings.shape[0] // dense_embeddings.shape[0],
            0,
        )

        # [B * M, num_outputs, N], [B * M, num_outputs]
        logits, iou_preds = self.mask_decoder(
            pc_embeddings,
            pc_pe,
            sparse_embeddings,
            dense_embeddings,
            aux_inputs=aux_inputs,
            multimask_output=multimask_output,
        )
        mask = logits > 0
        return mask, iou_preds, logits
