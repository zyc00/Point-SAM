from datasets import load_dataset
import numpy as np

dataset = load_dataset(
    "yuchen0187/pcmask",
    cache_dir="/home/ubuntu/projects/yuchen/fused_data",
    token="YOUR_HF_TOKEN",
    keep_in_memory=True,
    split="test",
)
mapping_points = []
mapping_masks = []
for i, data in enumerate(dataset):
    mapping_points.append(np.zeros(len(data["mask"])) + i)
    mapping_masks.append(np.arange(len(data["mask"])))
mapping_points = np.concatenate(mapping_points, axis=0)
mapping_masks = np.concatenate(mapping_masks, axis=0)
np.save("./mapping/points.npy", mapping_points)
np.save("./mapping/masks.npy", mapping_masks)
