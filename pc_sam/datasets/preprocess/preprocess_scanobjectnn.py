import numpy as np
import struct
import glob
import numpy as np
import sys

sys.path.append("/home/yuchen/workspace/pointcloud-sam")
from pc_sam.ply_utils import visualize_pc, save_ply

categories = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]
MAX_MASKS = 80
NUM_POINTS = 2048


def read_binary_point_cloud_part(bin_file_path):
    points = []
    with open(bin_file_path, "rb") as bin_file:
        data = bin_file.read()
    total_points = struct.unpack("f", data[:4])[0]
    start = 4
    for i in range(int(total_points)):
        try:
            point_data = struct.unpack("ff", data[start : start + 4 * 2])
            start += 4 * 2
            points.append(list(point_data))
        except:
            print("error")
    return points


def read_binary_point_cloud(bin_file_path):
    points = []
    with open(bin_file_path, "rb") as bin_file:
        data = bin_file.read()
    total_points = struct.unpack("f", data[:4])[0]
    start = 4
    for i in range(int(total_points)):
        try:
            point_data = struct.unpack("fffffffffff", data[start : start + 4 * 11])
            start += 4 * 11
            points.append(list(point_data))
        except:
            print("error")
    return points


for cat in categories:
    object_files = glob.glob(
        f"/mnt/data/ScanObjectNN/scanobjectnn_parts/{cat}/*part.bin"
    )

    xyz, rgb, valid, mask = (
        [],
        [],
        [],
        np.zeros([len(object_files), MAX_MASKS, NUM_POINTS]),
    )

    for i, f in enumerate(object_files):
        data = read_binary_point_cloud(f[:-9] + f[-4:])
        data = np.array(data)
        data_parts = read_binary_point_cloud_part(f)
        data_parts = np.array(data_parts)
        # filter_idx = data_parts[:, 1] != 0

        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        xyz.append(data[idx[:NUM_POINTS], 0:3])
        rgb.append(data[idx[:NUM_POINTS], 6:9])

        valid.append(len(np.unique(data_parts[idx[:NUM_POINTS], 0])))
        for j in range(valid[-1]):
            # if j > 30:
            #     print(f)
            mask[i, j] = (
                data_parts[idx[:NUM_POINTS], 0]
                == np.unique(data_parts[idx[:NUM_POINTS], 0])[j]
            )
    print(xyz[2].shape)
    print(xyz[2].min())
    print(xyz[2].max())
    visualize_pc("./test.ply", xyz[2], rgb[2] / 255)
    break


# data = read_binary_point_cloud(
#     "/mnt/data/ScanObjectNN/scanobjectnn_parts/bag/025_00011.bin"
# )
# data = np.array(data)
# print(np.unique(data[:, -1]))
# print(data[0])

# data = read_binary_point_cloud_part(
#     "/mnt/data/ScanObjectNN/scanobjectnn_parts/bag/025_00011_part.bin"
# )
# data = np.array(data)
# print(np.unique(data[:, -1]))
# print(data[0])
