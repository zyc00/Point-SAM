import numpy as np
import torch


def load_ply(filename):
    with open(filename, "r") as rf:
        while True:
            try:
                line = rf.readline()
            except:
                raise NotImplementedError
            if "end_header" in line:
                break
            if "element vertex" in line:
                arr = line.split()
                num_of_points = int(arr[2])

        # print("%d points in ply file" %num_of_points)
        points = np.zeros([num_of_points, 6])
        for i in range(points.shape[0]):
            point = rf.readline().split()
            assert len(point) == 6
            points[i][0] = float(point[0])
            points[i][1] = float(point[1])
            points[i][2] = float(point[2])
            points[i][3] = float(point[3])
            points[i][4] = float(point[4])
            points[i][5] = float(point[5])
        return points


def save_ply(mesh_path, points, rgb):
    """
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param mesh_path: File name to save
    :param points: [N, 3] array of points
    :param rgb: [N, 3] array of rgb values in the range [0~1]
    :return:
    """
    to_save = np.concatenate([points, rgb * 255], axis=-1)
    return np.savetxt(
        mesh_path,
        to_save,
        fmt="%.6f %.6f %.6f %d %d %d",
        comments="",
        header=(
            "ply\nformat ascii 1.0\nelement vertex {:d}\n"
            + "property float x\nproperty float y\nproperty float z\n"
            + "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            + "end_header"
        ).format(points.shape[0]),
    )


def visualize_prompts(path, points, prompt, labels, atol=0.005, points_num=1000):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(prompt, torch.Tensor):
        prompt = prompt.detach().cpu().numpy()

    # sample points around prompt
    for p in prompt:
        sampled_points = []
        for _ in range(points_num):
            diff = np.random.uniform(-atol, atol, [3])
            sampled_points.append(p + diff)
        sampled_points = np.stack(sampled_points)
        points = np.concatenate([points, sampled_points], axis=0)
    colors = np.ones_like(points)
    for i in range(prompt.shape[0]):
        start = -points_num * (len(prompt) - i)
        end = (
            -points_num * (len(prompt) - i - 1)
            if -points_num * (len(prompt) - i - 1) < 0
            else -1
        )
        colors[start:end] = [1, 0, 0] if labels[i] else [0, 1, 0]
    save_ply(path, points, colors)


def visualize_mask(path, points, mask):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    colors = np.ones_like(points)
    colors[mask > 0] = [1, 0, 0]
    save_ply(path, points, colors)


def visualize_pc(path, points, rgb=None):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if rgb is None:
        colors = np.ones_like(points)
    else:
        colors = rgb
    save_ply(path, points, colors)
