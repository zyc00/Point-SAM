from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Callable
import numpy as np
from pc_sam.ply_utils import visualize_mask, visualize_pc


def rotate_point_cloud(batch_data):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """batch_pc: BxNx3"""
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1):
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )
        Ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )
        Rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


class FuseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        local_dir: str = None,
        transform: Callable = None,
        token: str = None,
        split="partnet+shapenet",
        mask_batch: int = 1,
        augment: bool = True,
    ):
        self.dataset = load_dataset(
            data_path,
            cache_dir=local_dir,
            token=token,
            keep_in_memory=True,
            split=split,
        )
        # self.dataset = load_dataset(
        #     "parquet",
        #     data_files="data/partnet-00000-of-00008.parquet",
        # )["train"]
        self.dataset = self.dataset.with_format("np")
        self.mask_batch = mask_batch
        self.augment = augment
        if split == "test":
            self.augment = False
            self.fix_label = True
        else:
            self.fix_label = False

    def __getitem__(self, idx):
        points = self.dataset[idx]["xyz"].astype(np.float32)
        rgb = (self.dataset[idx]["rgb"] / 255).astype(np.float32)
        labels = self.dataset[idx]["mask"]

        # normalize points
        shift = np.mean(points, axis=0)
        scale = np.max(np.linalg.norm(points - shift, ord=2, axis=1))
        points = (points - shift) / scale

        # augment
        if self.augment:
            points = random_scale_point_cloud(points[None, ...])
            points = rotate_perturbation_point_cloud(points)
            points = rotate_point_cloud(points)
            # points = jitter_point_cloud(points)
            points = points.squeeze().astype(np.float32)

        # sample mask
        if self.fix_label:
            labels = np.stack([label for label in labels if label.sum() > 0])
            labels = labels[idx % len(labels)][None, :]
        else:
            labels = [
                label
                for label in labels
                if (label.sum() > 0 and label.sum() < label.shape[0] * 0.9)
            ]
            if len(labels) == 0:
                return self.__getitem__(idx + 1 % self.__len__())
            labels = np.stack(labels)

            num_masks = labels.shape[0]
            if num_masks < self.mask_batch:
                labels = np.repeat(labels, (self.mask_batch // num_masks + 1), 0)
                num_masks = labels.shape[0]
            label_idx = np.random.choice(num_masks, [self.mask_batch])
            labels = labels[label_idx]

        data = dict(points=points, rgb=rgb, seg_labels=labels)
        return data

    def __len__(self):
        return len(self.dataset)


class FuseDatasetVal(Dataset):
    def __init__(
        self,
        data_path: str,
        local_dir: str = None,
        transform: Callable = None,
        token: str = None,
        split="partnet+shapenet",
    ):
        self.dataset = load_dataset(
            data_path,
            cache_dir=local_dir,
            token=token,
            keep_in_memory=True,
            split=split,
        )
        self.mapping_points = np.load("./mapping/points.npy")
        self.mapping_masks = np.load("./mapping/masks.npy")

    def __getitem__(self, idx):
        points = np.array(self.dataset[int(self.mapping_points[idx])]["xyz"]).astype(
            np.float32
        )
        rgb = (
            np.array(self.dataset[int(self.mapping_points[idx])]["rgb"]).astype(
                np.float32
            )
            / 255
        )
        labels = np.array(self.dataset[int(self.mapping_points[idx])]["mask"])

        # normalize points
        shift = np.mean(points, axis=0)
        scale = np.max(np.linalg.norm(points - shift, ord=2, axis=1))
        points = (points - shift) / scale

        # sample mask
        labels = labels[self.mapping_masks[idx]][None, :]
        if labels.sum() == 0:
            return self.__getitem__(0)

        data = dict(points=points, rgb=rgb, seg_labels=labels)
        return data

    def __len__(self):
        return len(self.mapping_points)


if __name__ == "__main__":
    dataset = FuseDataset(
        data_path="yuchen0187/pcmask",
        local_dir="/home/ubuntu/projects/yuchen/fused_data",
        token="YOUR_HF_TOKEN",
    )
    data = dataset[0]
    visualize_mask("./test.ply", data["points"], data["seg_labels"])
