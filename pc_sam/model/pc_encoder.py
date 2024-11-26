# https://github.com/baaivision/Uni3D/blob/main/models/point_encoder.py
from typing import Union

import timm
import torch
import torch.nn as nn
from timm.models.eva import Eva
from timm.models.vision_transformer import VisionTransformer

from .common import KNNGrouper, NNGrouper, PatchEncoder


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


class PatchEmbedNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_patches) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_dim = hidden_dim or out_channels

        self.grouper = NNGrouper(num_patches)
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.blocks1 = nn.Sequential(
            *[Block(hidden_dim, hidden_dim, hidden_dim) for _ in range(3)]
        )
        self.blocks2 = nn.Sequential(
            *[Block(hidden_dim, hidden_dim, hidden_dim) for _ in range(3)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, coords: torch.tensor, features: torch.tensor):
        patches = self.grouper(coords, features)
        patch_features = patches["features"]  # [B, N, D]
        nn_idx = patches["nn_idx"]  # [B, N]

        x = self.in_proj(patch_features)
        x = self.blocks1(x)  # [B, N, D]
        y = x.new_zeros(x.shape[0], self.grouper.num_groups, x.shape[-1])
        y.scatter_reduce_(
            1, nn_idx.unsqueeze(-1).expand_as(x), x, "amax", include_self=False
        )
        x = self.blocks2(y)
        x = self.norm(x)
        x = self.out_proj(x)
        patches["embeddings"] = x
        return patches


class PatchEmbedHier(nn.Module):
    """PointNet++ style with hierarchical grouping."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_patches: list[int],
        patch_size: list[int],
        radius: list[float] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.grouper1 = KNNGrouper(
            num_patches[0],
            patch_size[0],
            radius=radius[0] if radius else None,
        )
        self.patch_encoder1 = PatchEncoder(in_channels, 128, [64, 128])

        self.grouper2 = KNNGrouper(
            num_patches[1],
            patch_size[1],
            radius=radius[1] if radius else None,
        )
        self.patch_encoder2 = PatchEncoder(128 + 3, out_channels, [128, 256])

    def forward(self, coords: torch.Tensor, features: torch.Tensor):
        patches1 = self.grouper1(coords, features)
        x1 = self.patch_encoder1(patches1["features"])
        patches1["embeddings"] = x1

        patches2 = self.grouper2(patches1["centers"], x1, use_fps=False)
        x2 = self.patch_encoder2(patches2["features"])
        patches2["embeddings"] = x2

        return [patches1, patches2]


def main():
    print(timm.list_models("vit_base*"))
    # print(timm.list_models("eva02*"))

    model_name = "vit_base_patch16_224"
    # model_name = "eva02_base_patch14_448"
    drop_path_rate = 0.2
    transformer = timm.create_model(
        model_name, pretrained=False, drop_path_rate=drop_path_rate
    )
    # patch_embed = PatchEmbed(6, 512, 512, 64)
    # patch_embed = PatchEmbedNN(7, 256, 512, 512)
    patch_embed = PatchEmbedHier(6, 512, [1024, 512], [32, 32])
    pc_encoder = PointCloudEncoder(
        patch_embed=patch_embed,
        transformer=transformer,
        embed_dim=512,
    ).cuda()

    points = torch.randn([2, 2048, 3]).cuda()
    colors = torch.rand_like(points)
    features, patches = pc_encoder(points, colors)
    print(features.shape)
    if isinstance(patches, list):
        for p in patches:
            print(p["features"].shape)
            print(p["centers"].shape)
    else:
        print(patches["features"].shape)
        print(patches["centers"].shape)


if __name__ == "__main__":
    main()
