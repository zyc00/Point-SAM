import sys

sys.path.append(".")
import glob

import torch
from datasets import Dataset
import argparse
import hydra
from omegaconf import OmegaConf
from accelerate.utils import set_seed, tqdm
from pc_sam.model.pc_sam import PointCloudSAM
from pc_sam.utils.torch_utils import replace_with_fused_layernorm
from pc_sam.model.loss import compute_iou
from safetensors.torch import load_model
import numpy as np
from scipy.spatial.transform import Rotation as R

r = R.from_euler("xyz", [-90, 180, 0], degrees=True)

valid_formats = {"ascii": "", "binary_big_endian": ">", "binary_little_endian": "<"}

ply_dtypes = dict(
    [
        (b"int8", "i1"),
        (b"char", "i1"),
        (b"uint8", "u1"),
        (b"uchar", "u1"),
        (b"int16", "i2"),
        (b"short", "i2"),
        (b"uint16", "u2"),
        (b"ushort", "u2"),
        (b"int32", "i4"),
        (b"int", "i4"),
        (b"uint32", "u4"),
        (b"uint", "u4"),
        (b"float32", "f4"),
        (b"float", "f4"),
        (b"float64", "f8"),
        (b"double", "f8"),
    ]
)


def save_colored_pc(file_name, xyz, rgb):
    # rgb is [0, 1]
    n = xyz.shape[0]
    f = open(file_name, "w")
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex %d\n" % n)
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    rgb = rgb * 255
    for i in range(n):
        if rgb.shape[1] == 3:
            f.write(
                "%f %f %f %d %d %d\n"
                % (xyz[i][0], xyz[i][1], xyz[i][2], rgb[i][0], rgb[i][1], rgb[i][2])
            )
        else:
            f.write(
                "%f %f %f %d %d %d\n"
                % (xyz[i][0], xyz[i][1], xyz[i][2], rgb[i][0], rgb[i][0], rgb[i][0])
            )


def normalize_colors(features, mean=0.5, std=0.5):
    features = features / 255
    if mean is not None:
        features = features - mean
    if std is not None:
        features = features / std
    return features


def normalize_points(points: np.ndarray):
    """Normalize the point cloud into a unit sphere."""
    assert points.ndim == 2 and points.shape[1] == 3, points.shape
    centroid = np.mean(points, axis=0)
    points = points - centroid
    norm = np.max(np.linalg.norm(points, ord=2, axis=1))
    return points / norm


def transform_fn(x):
    xyz = np.array(x["xyz"])
    rgb = np.array(x["rgb"])
    mask = np.array(x["mask"])

    # rgb[mask[1]] = [255, 0, 0]
    # save_colored_pc("test.ply", xyz, rgb / 255)
    # exit()

    # normalize
    xyz = normalize_points(xyz)
    rgb = normalize_colors(rgb)

    # to tensor
    xyz = torch.tensor(xyz, dtype=torch.float).cuda()
    rgb = torch.tensor(rgb, dtype=torch.float).cuda()
    mask = torch.tensor(mask, dtype=torch.bool).cuda()

    data = {
        "coords": xyz[None, ...],
        "features": rgb[None, ...],
        "gt_masks": mask[None, None, ...],
    }
    return data


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        if b"element" in line:
            line = line.split()
            num_points = int(line[2])

        elif b"property" in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        # Find point element
        if b"element vertex" in line:
            current_element = "vertex"
            line = line.split()
            num_points = int(line[2])

        elif b"element face" in line:
            current_element = "face"
            line = line.split()
            num_faces = int(line[2])

        elif b"property" in line:
            if current_element == "vertex":
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == "vertex":
                if not line.startswith("property list uchar int"):
                    raise ValueError("Unsupported faces property : " + line)

    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    with open(filename, "rb") as plyfile:
        # Check if the file start with ply
        if b"ply" not in plyfile.readline():
            raise ValueError("The file does not start whith the word ply")

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError("The file is not binary")

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:
            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [
                ("k", ext + "u1"),
                ("v1", ext + "i4"),
                ("v2", ext + "i4"),
                ("v3", ext + "i4"),
            ]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data["v1"], faces_data["v2"], faces_data["v3"])).T
            data = [vertex_data, faces]

        else:
            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def build_dataloader(dataset: Dataset):
    def transform_fn(x):
        xyz = np.array(x["xyz"])
        rgb = np.array(x["rgb"])
        mask = np.array(x["mask"])
        mask = np.stack(
            [
                mask[i]
                for i in range(mask.shape[0])
                if mask[i].sum() >= 25 and mask[i].sum() < 0.9 * mask.shape[1]
            ]
        )

        # rgb[mask[1]] = [255, 0, 0]
        # save_colored_pc("test.ply", xyz, rgb / 255)
        # exit()

        # normalize
        xyz = normalize_points(xyz)
        rgb = normalize_colors(rgb)

        # to tensor
        xyz = torch.tensor(xyz, dtype=torch.float).cuda()
        rgb = torch.tensor(rgb, dtype=torch.float).cuda()
        mask = torch.tensor(mask, dtype=torch.bool).cuda()

        data = [
            {
                "coords": xyz[None, ...],
                "features": rgb[None, ...],
                "gt_masks": mask[i][None, None, ...],
            }
            for i in range(mask.shape[0])
        ]
        return data

    dataloader = []
    for x in dataset:
        dataloader.append(transform_fn(x))
    return dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="large", help="path to config file"
    )
    parser.add_argument("--config_dir", type=str, default="../configs")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./pretrained/ours/mixture_10k_giant/model.safetensors",
    )
    args, unknown_args = parser.parse_known_args()

    # ---------------------------------------------------------------------------- #
    # Load configuration
    # ---------------------------------------------------------------------------- #
    with hydra.initialize(args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config, overrides=unknown_args)
        OmegaConf.resolve(cfg)
        # print(OmegaConf.to_yaml(cfg))

    seed = cfg.get("seed", 42)

    # ---------------------------------------------------------------------------- #
    # Setup model
    # ---------------------------------------------------------------------------- #
    set_seed(seed)
    model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
    model.apply(replace_with_fused_layernorm)

    # ---------------------------------------------------------------------------- #
    # Load pre-trained model
    # ---------------------------------------------------------------------------- #
    load_model(model, args.ckpt_path)

    # ---------------------------------------------------------------------------- #
    # Setup dataloader
    # ---------------------------------------------------------------------------- #
    partnet_mobility_dataset = glob.glob("/yuchen_slow/KITTI360/single/crops/*/*.ply")
    pbar = tqdm(total=len(partnet_mobility_dataset), miniters=10, maxinterval=60)

    # ---------------------------------------------------------------------------- #
    # Evaluate
    # ---------------------------------------------------------------------------- #
    model.eval()
    model.cuda()
    with torch.no_grad():
        total_ious = []
        object_ious = {}
        for pc in partnet_mobility_dataset:
            object_name = pc.split("/")[-1].split("_")[0]
            point_cloud = read_ply(pc)
            coords_full = np.column_stack(
                [point_cloud["x"], point_cloud["y"], point_cloud["z"]]
            ).astype(np.float32)
            coords_full = np.float32(r.apply(coords_full))
            colors_full = (
                np.column_stack([point_cloud["R"], point_cloud["G"], point_cloud["B"]])
            ).astype(np.float32)
            labels_full = point_cloud["label"].astype(np.int32)
            data = {"xyz": coords_full, "rgb": colors_full, "mask": labels_full}

            data = transform_fn(data)
            ious = [[] for _ in range(model.prompt_iters)]
            point_number = data["coords"].shape[1]
            # change fps number
            if point_number > 30000:
                model.pc_encoder.patch_embed.grouper.num_groups = 2048
                model.pc_encoder.patch_embed.grouper.group_size = 256
            else:
                model.pc_encoder.patch_embed.grouper.num_groups = min(
                    point_number, 2048
                )
                # model.pc_encoder.patch_embed.grouper.num_groups = 2048
                model.pc_encoder.patch_embed.grouper.group_size = 256
                if point_number < 256:
                    model.pc_encoder.patch_embed.grouper.group_size = 2
            outputs = model(**data, is_eval=True)
            gt_masks = data["gt_masks"].flatten(0, 1)
            for i_iter in range(len(outputs)):
                iou = (
                    compute_iou(outputs[i_iter]["prompt_masks"], gt_masks)
                    .detach()
                    .cpu()
                    .numpy()
                )
                ious[i_iter].append(iou)
            pbar.update(1)
            ious = np.array(ious).mean(axis=1)  # [prompt_iters]
            if object_ious.get(object_name) is None:
                object_ious[object_name] = []
            object_ious[object_name].append(ious)
            total_ious.append(ious)
            current_iou_mean = np.array(total_ious).mean(axis=0)
            print(f"Current mean IoU: {current_iou_mean}")
        total_ious = np.array(total_ious).mean(axis=0)
        print(f"Total mean IoU: {total_ious}")

        # average for each object
        for object_name, ious in object_ious.items():
            object_ious[object_name] = np.array(ious).mean(axis=0)

        # average for objects
        object_ious_mean = np.array(list(object_ious.values())).mean(axis=0)
        print(f"Object mean IoU: {object_ious_mean}")


if __name__ == "__main__":
    main()
