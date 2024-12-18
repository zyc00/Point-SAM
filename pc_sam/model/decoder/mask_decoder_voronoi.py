import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

# from pointnet2_ops import pointnet2_utils
from apex.normalization import FusedLayerNorm
from .common import knn_point


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            FusedLayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x) + x


class ResMlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            FusedLayerNorm(hidden_dim),
            nn.GELU(),
            *[ResBlock(hidden_dim, hidden_dim) for _ in range(num_layers)],
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class PropagateNN(nn.Module):
    def __init__(self, feats_dim, hidden_dim) -> None:
        super().__init__()
        self.mlp = ResMlp(feats_dim, hidden_dim, feats_dim, 3)
        # self.xyz_embed = ResMlp(4, hidden_dim, feats_dim, 0)
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            torch.randn((3, feats_dim // 2), dtype=torch.float32),
        )

    def forward(self, xyz, rgb, centers, center_feats):
        batch_size, num_points, _ = xyz.shape

        _, idx = knn_point(1, centers, xyz)
        idx = idx.squeeze(-1)

        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1)
            * centers.shape[1]
        )
        idx = idx + idx_base
        idx = idx.view(-1)  # B x N

        feats = center_feats.flatten(0, 1)[idx].view(batch_size, num_points, -1)
        neighborhood = xyz.flatten(0, 1) - centers.flatten(0, 1)[idx]
        neighborhood = neighborhood.view(batch_size, num_points, -1)

        # normalize relative position
        dist = torch.linalg.norm(
            neighborhood, dim=-1, keepdim=True, ord=2, dtype=xyz.dtype
        )
        neighborhood = neighborhood / (dist + 1e-8)
        neighborhood = neighborhood @ self.positional_encoding_gaussian_matrix
        neighborhood = 2 * torch.pi * neighborhood
        neighborhood = torch.cat(
            [torch.sin(neighborhood), torch.cos(neighborhood)], dim=-1
        )
        feats = self.mlp(feats + neighborhood)

        # xyz_feats = self.xyz_embed(torch.cat([neighborhood, dist], dim=-1))
        # feats = self.mlp(torch.cat([feats, neighborhood], dim=-1))  # B x N x C
        return feats


class MaskDecoderNN(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.output_upscaling = PropagateNN(transformer_dim, transformer_dim)

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        pc_embeddings: torch.Tensor,
        pc_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        centers: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            pc_embeddings=pc_embeddings,
            pc_pe=pc_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            xyz=xyz,
            rgb=rgb,
            centers=centers,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        pc_embeddings: torch.Tensor,
        pc_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        centers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(
            pc_embeddings, tokens.shape[0] // pc_embeddings.shape[0], dim=0
        )
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(
            pc_pe, tokens.shape[0] // pc_embeddings.shape[0], dim=0
        )
        xyz = torch.repeat_interleave(
            xyz, tokens.shape[0] // pc_embeddings.shape[0], dim=0
        )
        rgb = torch.repeat_interleave(
            rgb, tokens.shape[0] // pc_embeddings.shape[0], dim=0
        )
        centers = torch.repeat_interleave(
            centers, tokens.shape[0] // pc_embeddings.shape[0], dim=0
        )

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        upscaled_embedding = self.output_upscaling(xyz, rgb, centers, src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        masks = hyper_in @ upscaled_embedding.transpose(-1, -2)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
