from .shapenet_part import ShapeNetPartH5
from .partnet import PartNetH5
from .fuse_data import FuseDataset, FuseDatasetVal

import torch
from torch.utils.data import DataLoader

import numpy as np
import random


def build_dataloader(cfg, split="train"):
    def worker_init_fn(x):
        seed = cfg.seed + x
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        return

    if cfg.dataset == "ShapeNetPart":
        dataset = ShapeNetPartH5(cfg.datapath, split, cfg.num_points)
    elif cfg.dataset == "PartNet":
        dataset = PartNetH5(cfg.datapath, split, cfg.num_points)
    if cfg.dataset == "FuseData":
        if "val" in split:
            dataset = FuseDataset(
                cfg.datapath, cfg.localdir, token=cfg.token, split="test", mask_batch=1
            )
        else:
            dataset = FuseDataset(
                cfg.datapath,
                cfg.localdir,
                token=cfg.token,
                split=cfg.split,
                mask_batch=cfg.mask_batch,
            )
    else:
        raise NotImplementedError

    # build dataloader
    if "train" in split:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            worker_init_fn=worker_init_fn,
        )
    elif "val" in split:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.val_batch_size,
            shuffle=False,
            num_workers=24,
            worker_init_fn=worker_init_fn,
        )
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    return dataloader
