from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from torkit3d.nn.functional import batch_index_select

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


class MaskEncoderHier(nn.Module):
    """PointNet++ style with hierarchical grouping."""

    def __init__(self, embed_dim, in_channels=4, radius: list[float] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_channels = in_channels  # (x, y, z, logit)
        self.radius = radius

        self.patch_encoder1 = PatchEncoder(in_channels, 128, [64, 128])
        self.patch_encoder2 = PatchEncoder(128 + 3, embed_dim, [128, 256])

        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def forward(
        self,
        masks: Union[torch.Tensor, None],
        coords: torch.Tensor,
        centers1: torch.Tensor,
        knn_idx1: torch.Tensor,
        centers2: torch.Tensor,
        knn_idx2: torch.Tensor,
    ) -> torch.Tensor:
        if masks is None:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, 1, -1).expand(
                centers2.shape[0], centers2.shape[1], -1
            )
            return dense_embeddings
        else:
            masks = masks.detach()
            patches1 = group_with_centers_and_knn(
                coords,
                masks.unsqueeze(-1),
                centers1,
                knn_idx1,
                radius=self.radius[0] if self.radius else None,
            )
            x1 = self.patch_encoder1(patches1)

            patches2 = group_with_centers_and_knn(
                centers1,
                x1,
                centers2,
                knn_idx2,
                radius=self.radius[1] if self.radius else None,
            )
            x2 = self.patch_encoder2(patches2)
            return [x1, x2]


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x) + x


class ResMlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            *[ResBlock(hidden_dim, hidden_dim) for _ in range(num_layers)],
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class GroupNN(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, centers, feats, idx, mask_batch):
        batch_size, num_points, _ = xyz.shape
        neighborhood = xyz.flatten(0, 1) - centers.flatten(0, 1)[idx]  # [B, N, 3]
        neighborhood = neighborhood.view(batch_size, num_points, 3)
        neighborhood = torch.repeat_interleave(neighborhood, mask_batch, 0)

        # normalize relative position
        dist = torch.linalg.norm(
            neighborhood, dim=-1, keepdim=True, ord=2, dtype=xyz.dtype
        )
        neighborhood = neighborhood / (dist + 1e-8)

        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1) * self.num_group
        )
        idx = idx.view(batch_size, num_points) - idx_base
        idx_base = (
            torch.arange(0, batch_size * mask_batch, device=xyz.device).view(-1, 1)
            * self.num_group
        )
        idx = torch.repeat_interleave(idx, mask_batch, 0) + idx_base
        idx = idx.view(-1)
        neighborhood_feats = torch.cat(
            [feats.unsqueeze(-1), neighborhood, dist], dim=-1
        )
        return neighborhood_feats, idx


class MaskEncoderNN(nn.Module):
    def __init__(self, encoder_channel, num_group):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.num_group = num_group
        self.first_nn = nn.Linear(5, 1024)
        self.second_nn = ResMlp(1024, 1024, self.encoder_channel, 3)
        self.no_mask_embed = nn.Embedding(1, encoder_channel)

    def forward(self, point_groups, idx, centers, xyz):
        """
        point_groups : B N 4
        idx : B N
        -----------------
        feature_global : B G C
        """
        if point_groups is None:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, 1, -1).expand(
                centers.shape[0], centers.shape[1], -1
            )
            return dense_embeddings

        if point_groups is None:
            return torch.zeros(
                point_groups.shape[0], self.num_group, self.encoder_channel
            ).to(point_groups.device)

        point_groups = point_groups.unsqueeze(-1) 
        bs, n, _ = point_groups.shape

        # encoder
        nbr_xyz = xyz - batch_index_select(centers, idx, dim=1)  # [B, N, 3]
        dist = torch.linalg.norm(nbr_xyz, dim=-1, keepdim=True, ord=2)
        nbr_xyz = torch.repeat_interleave(nbr_xyz, point_groups.shape[0] // nbr_xyz.shape[0], 0)
        dist = torch.repeat_interleave(dist, point_groups.shape[0] // dist.shape[0], 0)
        idx = torch.repeat_interleave(idx, point_groups.shape[0] // idx.shape[0], 0)
        point_groups = torch.cat([point_groups, nbr_xyz, dist], dim=-1)
        feature = self.first_nn(point_groups)  # B N 256
        aggregrate_feature = torch.zeros(
            [bs * self.num_group, feature.shape[-1]],
            device=point_groups.device,
            dtype=feature.dtype,
        )
        aggregrate_feature = torch.scatter_reduce(
            aggregrate_feature,
            0,
            idx.unsqueeze(-1).expand(-1, feature.shape[-2], feature.shape[-1]).flatten(0, 1),
            feature.flatten(0, 1),
            "amax",
        )
        feature = self.second_nn(aggregrate_feature)  # B G 1024
        feature_global = feature.view(bs, self.num_group, -1)  # B G 1024
        return feature_global


class PromptEncoderNN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_group: int,
        group_size: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 2  # pos/neg point
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)

        # mask encoder
        self.group_divider = GroupNN(num_group, group_size)
        self.mask_encoder = MaskEncoderNN(embed_dim, num_group)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        point_embedding = self.pe_layer.forward_with_coords(points)
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def embed_masks(
        self,
        xyz: torch.Tensor,
        centers: torch.Tensor,
        masks: torch.Tensor,
        idx: torch.Tensor,
        mask_batch: int = 1,
    ) -> torch.Tensor:
        """Embeds mask inputs."""
        if masks is None:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, 1, -1).expand(
                centers.shape[0] * mask_batch, centers.shape[1], -1
            )
            return dense_embeddings
        masks = masks.detach()
        grouped_masks, idx = self.group_divider(xyz, centers, masks, idx, mask_batch)
        # grouped_masks: [BBM, N, 4], idx: [BBM, N]
        dense_embeddings = self.mask_encoder(grouped_masks, idx)
        return dense_embeddings
