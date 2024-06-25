import timm
from safetensors.torch import load_model
from .model import (
    PointCloudSAM,
    PointCloudEncoder,
    MaskEncoder,
    MaskDecoder,
    PatchEmbed,
    TwoWayTransformer,
)


def build_point_sam(checkpoint, num_group=512, group_size=64):
    patch_embed = PatchEmbed(6, 512, num_group, group_size)
    transformer = timm.create_model("eva02_large_patch14_448", pretrained=False)
    encoder = PointCloudEncoder(patch_embed, transformer, 256)
    mask_encoder = MaskEncoder(256)
    decoder_transformer = TwoWayTransformer(2, 256, 8, 2048)
    mask_decoder = MaskDecoder(256, decoder_transformer)
    model = PointCloudSAM(encoder, mask_encoder, mask_decoder)
    load_model(model, checkpoint)
    return model
