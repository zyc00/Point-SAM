import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

# from pointnet2_ops import pointnet2_utils
from apex.normalization import FusedLayerNorm
from .common import knn_point


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


class Propagate(nn.Module):
    def __init__(self, feats_dim, hidden_dim) -> None:
        super().__init__()
        self.relateve_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            FusedLayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feats_dim),
        )

        self.mlp = nn.Sequential(
            nn.Linear(feats_dim + 3, hidden_dim),
            FusedLayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feats_dim),
        )

        self.fc = nn.Sequential(
            nn.Linear(feats_dim, hidden_dim),
            FusedLayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feats_dim),
        )

    def forward(self, xyz, rgb, centers, center_feats):
        dist, idx = knn_point(3, centers, xyz)
        dist_recip = 1.0 / (dist**2 + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        rela_idx = (
            idx
            + torch.arange(centers.shape[0], device=xyz.device)[:, None, None]
            * centers.shape[1]
        ).flatten()
        rela_xyz = centers.flatten(0, 1)[rela_idx].reshape([xyz.shape[0], -1, 3, 3])
        rela_xyz = rela_xyz - xyz.unsqueeze(-2)  # [B, N, K, 3]
        rela_feats = self.relateve_mlp(rela_xyz)
        rela_feats = (rela_feats * weight[..., None]).sum(-2)  # [B, N, K]
        # weight    [B, N, K]
        # center feature    [B, N, K, D1]

        xyz_feats = center_feats.flatten(0, 1)[rela_idx].reshape(
            [xyz.shape[0], xyz.shape[1], 3, -1]
        )
        xyz_feats = xyz_feats * weight[..., None]
        xyz_feats = xyz_feats.sum(-2)  # [B, N, D]

        # center_feats = center_feats.transpose(-1, -2)
        # xyz_feats = pointnet2_utils.three_interpolate(
        #     center_feats.contiguous(), idx.int(), weight
        # )
        # xyz_feats = xyz_feats.transpose(-1, -2)  # B N C

        skip_feats = rela_feats + xyz_feats
        xyz_feats = torch.cat([skip_feats, rgb], dim=-1)
        xyz_feats = self.mlp(xyz_feats)
        xyz_feats = self.fc(skip_feats + xyz_feats)
        return xyz_feats


class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
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
        self.output_upscaling = Propagate(transformer_dim, transformer_dim)

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
        src = pc_embeddings
        src = src + dense_prompt_embeddings
        pos_src = pc_pe

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
