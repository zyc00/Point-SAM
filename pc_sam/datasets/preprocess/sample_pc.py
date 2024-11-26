import h5py
import numpy as np
import sys

import json

sys.path.append("/home/yuchen/workspace/pointcloud-sam")
from pc_sam.ply_utils import visualize_mask

data = h5py.File("/mnt/data/ins_seg_h5/Bed/train-00.h5")
print(np.array(data["label"]).max())
print(data.keys())
print(np.array(data["pts"]).min())
# for label in np.unique(np.array(data["label"])[0]):
#     #     print(i)
#     #     assert np.array(data["gt_mask"])[0, i].sum() == 0
#     visualize_mask(
#         f"./partnet_{label}.ply",
#         np.array(data["pts"])[0],
#         np.array(data["label"])[0] == label,
#     )

# with open("/mnt/data/ins_seg_h5_for_detection/Bed-3/train-00.json") as f:
#     jsfile = json.load(f)
# print(len(jsfile[0]["ins_seg"]))
