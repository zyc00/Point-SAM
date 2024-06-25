from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from .common import PatchEncoder, group_with_centers_and_knn


# https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/prompt_encoder.py
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [-1,1]."""
        # assuming coords are in [-1, 1] and have d_1 x ... x d_n x D shape
        coords = coords @ self.positional_encoding_gaussian_matrix
        # TODO: Why using 2 * np.pi?
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: shape (..., coord_dim), normalized coordinates in [-1, 1].

        Returns:
            torch.Tensor: shape (..., num_pos_feats), positional encoding.
        """
        if (coords < -1 - 1e-6).any() or (coords > 1 + 1e-6).any():
            print("Bounds: ", (coords.min(), coords.max()))
            raise ValueError(f"Input coordinates must be normalized to [-1, 1].")
        # TODO: whether to convert to float?
        return self._pe_encoding(coords)


class PointEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 2  # pos/neg point
        point_embeddings = [
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)

    def forward(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Embeds point prompts.

        Args:
            points: [..., 3]. Point coordinates.
            labels: [...], integer (or boolean). Point labels.

        Returns:
            torch.Tensor: [..., embed_dim]. Embedded points.
        """
        assert points.shape[:-1] == labels.shape
        point_embedding = self.pe_layer.forward(points)
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding


class MaskEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        in_channels=4,
        radius=None,
        centralize_features=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_channels = in_channels  # (x, y, z, logit)
        self.radius = radius
        self.centralize_features = centralize_features

        self.patch_encoder = PatchEncoder(in_channels, embed_dim, [128, 512])
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def forward(
        self,
        masks: Union[torch.Tensor, None],
        coords: torch.Tensor,
        centers: torch.Tensor,
        knn_idx: torch.Tensor,
        center_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Embeds mask inputs.

        Args:
            masks: [B * M, N], float. Mask inputs.
            coords: [B, N, 3]. Point coordinates.
            centers: [B, L, 3]. Center coordinates.
            knn_idx: [B, L, K]. KNN indices.
            center_idx: [B, L]. Index of center in the point cloud.

        Returns:
            torch.Tensor: [B * M, L, embed_dim]. Dense embeddings.
        """
        if masks is None:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, 1, -1).expand(
                centers.shape[0], centers.shape[1], -1
            )
        else:
            masks = masks.detach()
            patches = group_with_centers_and_knn(
                coords,
                masks.unsqueeze(-1),
                centers,
                knn_idx,
                radius=self.radius,
                center_idx=center_idx,
                centralize_features=self.centralize_features,
            )
            dense_embeddings = self.patch_encoder(patches)
        return dense_embeddings
