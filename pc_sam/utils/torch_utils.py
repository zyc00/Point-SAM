import random

import numpy as np
import torch
from torch import nn


def worker_init_fn(worker_id: int, rank: int = 0):
    # https://github.com/Lightning-AI/pytorch-lightning/blob/0f12271d7feeacb6fbe5d70d2ce057da4a04d8b4/src/lightning/fabric/utilities/seed.py#L78
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    global_rank = rank
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (
        stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]
    ).sum()
    random.seed(stdlib_seed)


def replace_with_fused_layernorm(module: nn.Module):
    # https://github.com/huggingface/pytorch-image-models/pull/1674/files
    from apex.normalization import FusedLayerNorm

    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            # "memory_efficient" option can not be used for training as it will modify variables in-place.
            fused_layernorm = FusedLayerNorm(
                child.normalized_shape, child.eps, child.elementwise_affine
            )
            module.register_module(name, fused_layernorm)
