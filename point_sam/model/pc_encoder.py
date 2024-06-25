# https://github.com/baaivision/Uni3D/blob/main/models/point_encoder.py
from typing import Union

import timm
import torch
import torch.nn as nn
from timm.models.eva import Eva
from timm.models.vision_transformer import VisionTransformer

from .common import KNNGrouper, PatchEncoder


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_patches,
        patch_size,
        radius: float = None,
        centralize_features=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.grouper = KNNGrouper(
            num_patches,
            patch_size,
            radius=radius,
            centralize_features=centralize_features,
        )

        self.patch_encoder = PatchEncoder(in_channels, out_channels, [128, 512])

    def forward(self, coords: torch.Tensor, features: torch.Tensor):
        patches = self.grouper(coords, features)
        patch_features = patches["features"]  # [B, L, K, C_in]
        x = self.patch_encoder(patch_features)
        patches["embeddings"] = x
        return patches


class PatchDropout(nn.Module):
    """Randomly drop patches.

    References:
    - https://arxiv.org/abs/2212.00794
    - `timm.layers.patch_dropout`. It uses `argsort` rather than `topk`, which might be inefficient.
    """

    def __init__(self, prob, num_prefix_tokens: int = 1):
        super().__init__()
        assert 0.0 <= prob < 1.0, prob
        self.prob = prob
        # exclude CLS token (or other prefix tokens)
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, x: torch.Tensor):
        # x: [B, L, ...]
        if not self.training or self.prob == 0.0:
            return x

        if self.num_prefix_tokens:
            prefix_tokens = x[:, : self.num_prefix_tokens]
            x = x[:, self.num_prefix_tokens :]
        else:
            prefix_tokens = None

        B, L = x.shape[:2]
        num_keep = max(1, int(L * (1.0 - self.prob)))
        rand = torch.randn(B, L, device=x.device)
        keep_indices = rand.topk(num_keep, dim=1).indices
        _keep_indices = keep_indices.reshape((B, num_keep) + (-1,) * (x.dim() - 2))
        _keep_indices = _keep_indices.expand((-1, -1) + x.shape[2:])
        x = x.gather(1, _keep_indices)

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        return x


class PointCloudEncoder(nn.Module):
    def __init__(
        self,
        patch_embed: PatchEmbed,
        transformer: Union[VisionTransformer, Eva],
        embed_dim: int,
        patch_drop_rate=0.0,
    ):
        super().__init__()
        self.transformer_dim = transformer.embed_dim
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = patch_embed
        # Project patch features to transformer input dim
        self.patch_proj = nn.Linear(self.patch_embed.out_channels, self.transformer_dim)

        # Positional embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.transformer_dim)
        )

        assert patch_drop_rate == 0, "PatchDropout is not compatible with decoder."
        if patch_drop_rate > 0:
            self.patch_dropout = PatchDropout(patch_drop_rate, num_prefix_tokens=0)
        else:
            self.patch_dropout = nn.Identity()

        # Transformer encoder
        self.transformer = transformer

        # Project transformer output to embedding dim
        self.out_proj = nn.Linear(self.transformer_dim, self.embed_dim)

    def forward(self, coords, features):
        # Group points into patches and get embeddings
        patches = self.patch_embed(coords, features)
        if isinstance(patches, list):
            patch_embed = patches[-1]["embeddings"]
            centers = patches[-1]["centers"]
        else:
            patch_embed = patches["embeddings"]  # [B, L, D]
            centers = patches["centers"]  # [B, L, 3]
        patch_embed = self.patch_proj(patch_embed)

        # Positional embedding for patches
        pos_embed = self.pos_embed(centers)
        x = patch_embed + pos_embed

        # Dropout patch
        x = self.patch_dropout(x)
        # Dropout features
        x = self.transformer.pos_drop(x)

        for block in self.transformer.blocks:
            x = block(x)
        # In fact, only norm or fc_norm is not identity in those transformers.
        x = self.transformer.norm(x)
        x = self.transformer.fc_norm(x)
        x = self.out_proj(x)

        return x, patches


class Block(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        # Follow timm.layers.mlp
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # PreLN. Follow timm.models.vision_transformer
        return x + self.mlp(self.norm(x))
