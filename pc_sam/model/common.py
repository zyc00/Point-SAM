# https://github.com/baaivision/Uni3D/blob/main/models/point_encoder.py
from typing import Union

import torch
from torch import nn
from torch.nn import functional as F
from torkit3d.nn.functional import batch_index_select
from torkit3d.ops.sample_farthest_points import sample_farthest_points
from torkit3d.ops.chamfer_distance import chamfer_distance


def fps(points: torch.Tensor, num_samples: int):
    """A wrapper of farthest point sampling (FPS).

    Args:
        points: [B, N, 3]. Input point clouds.
        num_samples: int. The number of points to sample.

    Returns:
        torch.Tensor: [B, num_samples, 3]. Sampled points.
    """
    idx = sample_farthest_points(points, num_samples)
    sampled_points = batch_index_select(points, idx, dim=1)
    return sampled_points


def knn_points(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    sorted: bool = False,
    transpose: bool = False,
):
    """Compute k nearest neighbors.

    Args:
        query: [B, N1, D], query points. [B, D, N1] if @transpose is True.
        key:  [B, N2, D], key points. [B, D, N2] if @transpose is True.
        k: the number of nearest neighbors.
        sorted: whether to sort the results
        transpose: whether to transpose the last two dimensions.

    Returns:
        torch.Tensor: [B, N1, K], distances to the k nearest neighbors in the key.
        torch.Tensor: [B, N1, K], indices of the k nearest neighbors in the key.
    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    # Compute pairwise distances, [B, N1, N2]
    distance = torch.cdist(query, key)
    if k == 1:
        knn_dist, knn_ind = torch.min(distance, dim=2, keepdim=True)
    else:
        knn_dist, knn_ind = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)
    return knn_dist, knn_ind


class KNNGrouper(nn.Module):
    """Group points based on K nearest neighbors.

    A number of points are sampled as centers by farthest point sampling (FPS).
    Each group is formed by the center and its k nearest neighbors.
    """

    def __init__(self, num_groups, group_size, radius=None, centralize_features=False):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.radius = radius
        self.centralize_features = centralize_features

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, use_fps=True):
        """
        Args:
            xyz: [B, N, 3]. Input point clouds.
            features: [B, N, C]. Point features.
            use_fps: bool. Whether to use farthest point sampling.
                If not, `xyz` should already be sampled by FPS.

        Returns:
            dict: {
                features: [B, G, K, 3 + C]. Group features.
                centers: [B, G, 3]. Group centers.
                knn_idx: [B, G, K]. The indices of k nearest neighbors.
            }
        """
        batch_size, num_points, _ = xyz.shape
        with torch.no_grad():
            if use_fps:
                fps_idx = sample_farthest_points(xyz.float(), self.num_groups)
                centers = batch_index_select(xyz, fps_idx, dim=1)
            else:
                fps_idx = torch.arange(self.num_groups, device=xyz.device)
                fps_idx = fps_idx.expand(batch_size, -1)
                centers = xyz[:, : self.num_groups]
            _, knn_idx = knn_points(centers, xyz, self.group_size)  # [B, G, K]

        batch_offset = torch.arange(batch_size, device=xyz.device) * num_points
        batch_offset = batch_offset.reshape(-1, 1, 1)
        knn_idx_flat = (knn_idx + batch_offset).reshape(-1)  # [B * G * K]

        nbr_xyz = xyz.reshape(-1, 3)[knn_idx_flat]
        nbr_xyz = nbr_xyz.reshape(batch_size, self.num_groups, self.group_size, 3)
        nbr_xyz = nbr_xyz - centers.unsqueeze(2)  # [B, G, K, 3]
        # NOTE: Follow PointNext to normalize the relative position
        if self.radius is not None:
            nbr_xyz = nbr_xyz / self.radius

        nbr_feats = features.reshape(-1, features.shape[-1])[knn_idx_flat]
        nbr_feats = nbr_feats.reshape(
            batch_size, self.num_groups, self.group_size, features.shape[-1]
        )

        group_feats = [nbr_xyz, nbr_feats]
        if self.centralize_features:
            center_feats = batch_index_select(features, fps_idx, dim=1)
            group_feats.append(nbr_feats - center_feats.unsqueeze(2))

        group_feats = torch.cat(group_feats, dim=-1)
        return dict(
            features=group_feats, centers=centers, knn_idx=knn_idx, fps_idx=fps_idx
        )


def group_with_centers_and_knn(
    xyz: torch.Tensor,
    features: torch.Tensor,
    centers: torch.Tensor,
    knn_idx: torch.Tensor,
    radius: float = None,
    centralize_features: bool = False,
    center_idx: torch.Tensor = None,
):
    """Group points based on K nearest neighbors.

    Args:
        xyz: [B, N, 3]. Input point clouds.
        features: [B * M, N, C]. Point features. Support multiple features for the same point cloud.
        centers: [B, L, 3]. Group centers.
        knn_idx: [B, L, K]. The indices of k nearest neighbors.

    Returns:
        torch.Tensor: [B * M, L, K, 3 + C]. Group features.
    """
    assert xyz.dim() == features.dim(), (xyz.shape, features.shape)
    assert xyz.shape[1] == features.shape[1], (xyz.shape, features.shape)
    assert xyz.shape[0] == centers.shape[0] == knn_idx.shape[0]
    assert knn_idx.shape[:2] == centers.shape[:2], (knn_idx.shape, centers.shape)

    # 1. Compute neighborhood coordinates
    batch_size, num_points, _ = xyz.shape
    _, num_patches, patch_size = knn_idx.shape

    batch_offset = torch.arange(batch_size, device=xyz.device) * num_points
    batch_offset = batch_offset.reshape(-1, 1, 1)
    knn_idx_flat = (knn_idx + batch_offset).reshape(-1)  # [B * L * K]

    nbr_xyz = xyz.reshape(-1, 3)[knn_idx_flat]
    nbr_xyz = nbr_xyz.reshape(batch_size, num_patches, patch_size, 3)
    nbr_xyz = nbr_xyz - centers.unsqueeze(2)  # [B, L, K, 3]
    if radius is not None:
        # dist = torch.linalg.norm(nbr_xyz, dim=-1, ord=2)
        # print(dist.max(), dist.min(), dist.mean())
        nbr_xyz = nbr_xyz / radius

    # 2. Compute neighborhood features
    batch_size2 = features.shape[0]
    repeats = features.shape[0] // xyz.shape[0]
    knn_idx2 = torch.repeat_interleave(knn_idx, repeats, dim=0)  # [B*M,L,K]

    batch_offset = torch.arange(batch_size2, device=xyz.device) * num_points
    batch_offset = batch_offset.reshape(-1, 1, 1)
    knn_idx_flat = (knn_idx2 + batch_offset).reshape(-1)  # [B*M*L*K]
    nbr_feats = features.reshape(-1, features.shape[-1])[knn_idx_flat]
    nbr_feats = nbr_feats.reshape(
        batch_size2, num_patches, patch_size, features.shape[-1]
    )

    # 3. Concatenate features
    nbr_xyz = torch.repeat_interleave(nbr_xyz, repeats, dim=0)
    group_feats = [nbr_xyz, nbr_feats]
    if centralize_features:
        center_idx = torch.repeat_interleave(center_idx, repeats, dim=0)
        center_feats = batch_index_select(features, center_idx, dim=1)
        group_feats.append(nbr_feats - center_feats.unsqueeze(2))
    return torch.cat(group_feats, dim=-1)


class NNGrouper(nn.Module):
    """Group points based on the nearest neighbors."""

    def __init__(self, num_groups: int):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        with torch.no_grad():
            fps_idx = sample_farthest_points(xyz.float(), self.num_groups)
            centers = batch_index_select(xyz, fps_idx, dim=1)
            _, nn_idx = knn_points(xyz, centers, 1)  # [B, N, 1]

        # Compute the relative position of each point to its nearest center
        nn_idx = nn_idx.squeeze(-1)
        nbr_xyz = xyz - batch_index_select(centers, nn_idx, dim=1)  # [B, N, 3]

        # Normalize the relative position
        dist = torch.linalg.norm(nbr_xyz, dim=-1, keepdim=True, ord=2)
        nbr_xyz = nbr_xyz / torch.clamp(dist, min=1e-8)

        group_feats = torch.cat([nbr_xyz, dist, features], dim=-1)
        return dict(features=group_feats, centers=centers, nn_idx=nn_idx)

def group_with_centers_and_nn(
    xyz: torch.Tensor,
    features: torch.Tensor,
    centers: torch.Tensor,
    nn_idx: torch.Tensor,
):
    """
    Group points based on the voronoi diagram.

    Args:
        xyz: [B, N, 3]. Input point clouds.
        features: [B, N, C]. Point features.
        centers: [B, L, 3]. Group centers.
        nn_idx: [B, N]. The indices of the nearest neighbors.

    Returns:
        torch.Tensor: [B, L, 3 + C]. Group features.
    """
    nbr_xyz = xyz - batch_index_select(centers, nn_idx, dim=1)  # [B, N, 3]
    dist = torch.linalg.norm(nbr_xyz, dim=-1, keepdim=True, ord=2)
    nbr_xyz = nbr_xyz / torch.clamp(dist, min=1e-8)
    group_feats = torch.cat([nbr_xyz, dist, features], dim=-1)
    return group_feats

def compute_interp_weights(query: torch.Tensor, key: torch.Tensor, k=3, eps=1e-8):
    """Compute interpolation weights for each query point.

    Args:
        query: [B, Nq, 3]. Query points.
        key: [B, Nk, 3]. Key points.
        k: int. The number of nearest neighbors.
        eps: float. A small value to avoid division by zero.

    Returns:
        torch.Tensor: [B, Nq, K], indices of the k nearest neighbors in the key.
        torch.Tensor: [B, Nq, K], interpolation weights.
    """
    dist, idx = knn_points(query, key, k)
    inv_dist = 1.0 / torch.clamp(dist.square(), min=eps)
    normalizer = torch.sum(inv_dist, dim=2, keepdim=True)
    weight = inv_dist / normalizer  # [B, Nq, K]
    return idx, weight


def interpolate_features(x: torch.Tensor, index: torch.Tensor, weight: torch.Tensor):
    """
    Interpolates features based on the given index and weight.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_keys, num_features).
        index (torch.Tensor): The index tensor of shape (batch_size, num_queries, K).
        weight (torch.Tensor): The weight tensor of shape (batch_size, num_queries, K).

    Returns:
        torch.Tensor: The interpolated features tensor of shape (batch_size, num_queries, num_features).
    """
    B, Nq, K = index.shape
    batch_offset = torch.arange(B, device=x.device).reshape(-1, 1, 1) * x.shape[1]
    index_flat = (index + batch_offset).flatten()  # [B*Nq*K]
    _x = x.flatten(0, 1)[index_flat].reshape(B, Nq, K, x.shape[-1])
    return (_x * weight.unsqueeze(-1)).sum(-2)


def repeat_interleave(x: torch.Tensor, repeats: int, dim: int):
    if repeats == 1:
        return x
    shape = list(x.shape)
    shape.insert(dim + 1, 1)
    shape[dim + 1] = repeats
    x = x.unsqueeze(dim + 1).expand(shape).flatten(dim, dim + 1)
    return x


@torch.no_grad()
def sample_prompts_adapter(
    points: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_logits: Union[torch.Tensor, None],
    threshold: float = None,
    is_eval = False,
):
    """Select prompt sampler based on iou."""
    if pred_logits is None:
        return sample_fixed_points(
            points, gt_masks, pred_logits, threshold, from_error_region=True
        )
    else:
        batch_size, num_masks, _ = gt_masks.shape

        # if the batch iou is less than 0.5, use fixed sampler
        gt_masks_copy = gt_masks.reshape(batch_size * num_masks, -1)
        if threshold is None:
            pred_masks = pred_logits > 0
        else:
            pred_masks = pred_logits.sigmoid() > threshold

        iou = (gt_masks_copy & pred_masks).sum() / (gt_masks_copy | pred_masks).sum()
        if iou < 1 or is_eval:
            return sample_fixed_points(
                points, gt_masks, pred_logits, threshold, from_error_region=False
            )
        else:
            return sample_prompts(points, gt_masks, pred_logits, threshold)


@torch.no_grad()
def sample_prompts(
    points: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_logits: Union[torch.Tensor, None],
    threshold: float = None,
):
    """Sample prompts from point clouds given ground-truth and predicted masks.

    Args:
        points: [B, N, 3]. Input point clouds.
        gt_masks: [B, M, N], bool. Ground-truth (binary) masks.
        pred_logits: A float tensor of shape [B*M, N]. Predicted logits.
            If None, the prompt points will be sampled from the ground-truth masks.

    Returns:
        torch.Tensor: [B*M, 1, 3]. Prompt points.
        torch.Tensor: [B*M, 1], bool. Prompt labels.
    """
    batch_size, num_masks, _ = gt_masks.shape

    # The prompt point will be sampled from the error region.
    if pred_logits is None:
        diff_masks = gt_masks
    else:
        pred_logits = pred_logits.reshape(batch_size, num_masks, -1)
        assert gt_masks.shape == pred_logits.shape, (gt_masks.shape, pred_logits.shape)
        if threshold is None:
            pred_masks = pred_logits > 0
        else:
            pred_masks = pred_logits.sigmoid() > threshold
        diff_masks = gt_masks != pred_masks

    prompt_coords, prompt_labels = [], []
    for i in range(batch_size):
        for j in range(num_masks):
            diff_inds = torch.nonzero(diff_masks[i, j])  # [?, 1]
            if len(diff_inds) == 0:
                diff_inds = torch.nonzero(gt_masks[i, j])
            diff_inds = diff_inds.squeeze(1)  # [?]
            idx = diff_inds[torch.randint(0, len(diff_inds), [1])]
            prompt_coords.append(points[i][idx])
            prompt_labels.append(gt_masks[i, j][idx])

    prompt_coords = torch.stack(prompt_coords)
    prompt_labels = torch.stack(prompt_labels)
    return prompt_coords, prompt_labels


@torch.no_grad()
def sample_fixed_points(
    points: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_logits: Union[torch.Tensor, None],
    threshold: float = None,
    from_error_region: bool = False,
):
    """Sample prompts from point clouds given ground-truth and predicted masks.

    Args:
        points: [B, N, 3]. Input point clouds.
        gt_masks: [B, M, N], bool. Ground-truth (binary) masks.
        pred_logits: A float tensor of shape [B*M, N]. Predicted logits.
            If None, the prompt points will be sampled from the ground-truth masks.

    Returns:
        torch.Tensor: [B*M, 1, 3]. Prompt points.
        torch.Tensor: [B*M, 1], bool. Prompt labels.
    """
    batch_size, num_masks, _ = gt_masks.shape

    # The prompt point will be sampled from the error region.
    if pred_logits is None:
        fn = gt_masks
        fp = torch.zeros_like(fn)
    else:
        pred_logits = pred_logits.reshape(batch_size, num_masks, -1)
        assert gt_masks.shape == pred_logits.shape, (gt_masks.shape, pred_logits.shape)
        if threshold is None:
            pred_masks = pred_logits > 0
        else:
            pred_masks = pred_logits.sigmoid() > threshold
        fn = gt_masks & ~pred_masks
        fp = ~gt_masks & pred_masks

    prompt_points, prompt_labels = [], []
    if from_error_region:
        mask = fn | fp
        for i in range(batch_size):
            for j in range(num_masks):
                coords, label, _ = sample_furthest_points_from_border(
                    points[i], mask[i, j], gt_masks[i, j]
                )
                prompt_points.append(coords)
                prompt_labels.append(label)
    else:
        for i in range(batch_size):
            for j in range(num_masks):
                pprompt_coord, pprompt_label, pdist = (
                    sample_furthest_points_from_border(
                        points[i], fn[i, j], gt_masks[i, j]
                    )
                )
                nprompt_coord, nprompt_label, ndist = (
                    sample_furthest_points_from_border(
                        points[i], fp[i, j], gt_masks[i, j]
                    )
                )
                if pdist > ndist:
                    prompt_points.append(pprompt_coord)
                    prompt_labels.append(pprompt_label)
                elif ndist == -1:
                    pprompt_coord, pprompt_label, pdist = (
                        sample_furthest_points_from_border(
                            points[i], gt_masks[i, j], gt_masks[i, j]
                        )
                    )
                    prompt_points.append(pprompt_coord)
                    prompt_labels.append(pprompt_label)
                else:
                    prompt_points.append(nprompt_coord)
                    prompt_labels.append(nprompt_label)

    prompt_points = torch.stack(prompt_points)
    prompt_labels = torch.stack(prompt_labels)
    return prompt_points, prompt_labels


def sample_furthest_points_from_border(
    coords: torch.Tensor, lables: torch.Tensor, gt: torch.Tensor
):
    """
    Sample points from the border of the mask.

    Args:
        coords: [N, 3]. Input point clouds.
        lables: [N]. Point labels.
        gt: [N]. Ground-truth labels.
    """
    bg_inds = lables == 0
    fg_inds = lables == 1

    # if bg_inds or fg_inds is empty, return None
    if bg_inds.sum() == 0 or fg_inds.sum() == 0:
        return None, None, -1

    # All distances from foreground points to background points
    min_dists, _ = chamfer_distance(coords[fg_inds][None, ...], coords[bg_inds][None, ...])

    # Sample the farthest points from the border
    center_idx = torch.argmax(min_dists)
    center_coords = coords[fg_inds][center_idx]
    center_dist = torch.max(min_dists)
    center_label = gt[fg_inds][center_idx]

    return center_coords[None, ...], center_label[None, ...], center_dist


class PatchEncoder(nn.Module):
    """Encode point patches following the PointNet structure for segmentation."""

    def __init__(self, in_channels, out_channels, hidden_dims: list[int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # NOTE: The original Uni3D implementation uses BatchNorm1d, while we use LayerNorm.
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], out_channels),
        )

    def forward(self, point_patches: torch.Tensor):
        # point_patches: [B, L, K, C_in]
        x = self.conv1(point_patches)
        y = torch.max(x, dim=-2, keepdim=True).values
        x = torch.cat([y.expand_as(x), x], dim=-1)
        x = self.conv2(x)  # [B, L, K, C_out]
        y = torch.max(x, dim=-2).values  # [B, L, C_out]
        return y
    
class PatchEncoderNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims: list[int]) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], out_channels),
        )

    def forward(self, point_patches: torch.Tensor, nn_idx: torch.Tensor, center_number: int) -> torch.Tensor:
        # point_patches: [B, N, C_in]
        x = self.conv1(point_patches)
        y = torch.zeros([x.shape[0], center_number, x.shape[-1]], device=x.device, dtype=x.dtype)
        y = torch.scatter_reduce(y, 1, nn_idx, x, "max")
        x_max = torch.zeros_like(x)
        x_max = torch.gather(y, 1, nn_idx.unsqueeze(-1).expand_as(y))
        x = torch.cat([x_max, x], dim=-1)
        x = self.conv2(x)
        y = torch.zeros([x.shape[0], center_number, x.shape[-1]], device=x.device, dtype=x.dtype)
        y = torch.scatter_reduce(y, 1, nn_idx, x, "max")
        return y
