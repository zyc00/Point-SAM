import timm
import torch
import torch.nn as nn
from apex.normalization import FusedLayerNorm

from .encoder.pc_encoder import PointcloudEncoderNN
from .encoder.prompt_encoder import PromptEncoderNN
from .decoder.mask_decoder_voronoi import MaskDecoderNN
from .encoder.pc_encoder import PointCloudEncoder
from .encoder.prompt_encoder import PromptEncoder
from .decoder.mask_decoder_trm import MaskDecoder
from .transformer import TwoWayTransformer
from .pc_sam import PointCloudSAM


def fn(module: nn.Module):
    for n, m in module.named_children():
        if isinstance(m, nn.LayerNorm):
            module.register_module(
                n, FusedLayerNorm(m.normalized_shape, m.eps, m.elementwise_affine)
            )


def build_sam(cfg):
    point_transformer = timm.create_model(
        cfg.encoder_model_name, drop_path_rate=cfg.drop_path_rate
    )
    point_transformer.apply(fn)
    pc_encoder = PointcloudEncoderNN(
        point_transformer,
        cfg.encoder_trans_dim,
        cfg.encoder_embed_dim,
        cfg.encoder_group_size,
        cfg.encoder_num_group,
        cfg.encoder_pc_encoder_dim,
        cfg.encoder_patch_dropout,
    )
    prompt_encoder = PromptEncoderNN(
        cfg.prompt_embed_dim, cfg.encoder_num_group, cfg.encoder_group_size
    )
    two_way_transformer = TwoWayTransformer(
        cfg.decoder_trans_depth,
        cfg.decoder_embed_dim,
        cfg.decoder_num_head,
        cfg.decoder_mlp_dim,
    )
    mask_decoder = MaskDecoderNN(cfg.decoder_trans_dim, two_way_transformer)
    sam = PointCloudSAM(pc_encoder, prompt_encoder, mask_decoder, cfg.prompt_iters)
    return sam


# unit test
if __name__ == "__main__":
    point_transformer: nn.Module = timm.create_model("eva02_large_patch14_448")
    point_transformer.apply(fn)
    print(torch.backends.cuda.flash_sdp_enabled())
    # True
    print(torch.backends.cuda.mem_efficient_sdp_enabled())
    # True
    print(torch.backends.cuda.math_sdp_enabled())
    print(point_transformer)
