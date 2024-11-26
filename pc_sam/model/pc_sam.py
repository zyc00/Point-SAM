"""Segment Anything Model for Point Clouds.

References:
- https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/sam.py
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torkit3d.nn.functional import batch_index_select

from .common import repeat_interleave, sample_prompts, sample_prompts_adapter
from .mask_decoder import AuxInputs, MaskDecoder
from .pc_encoder import PointCloudEncoder
from .prompt_encoder import MaskEncoder, PointEncoder


class PointCloudSAM(nn.Module):
    def __init__(
        self,
        pc_encoder: PointCloudEncoder,
        mask_encoder: MaskEncoder,
        mask_decoder: MaskDecoder,
        prompt_iters: int,
        enable_mask_refinement_iterations=True,
    ):
        super().__init__()
        self.pc_encoder = pc_encoder
        self.point_encoder = PointEncoder(pc_encoder.embed_dim)
        self.mask_encoder = mask_encoder
        self.mask_decoder = mask_decoder
        self.prompt_iters = prompt_iters
        self.enable_mask_refinement_iterations = enable_mask_refinement_iterations

    def predict_masks(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        prompt_coords: torch.Tensor,
        prompt_labels: torch.Tensor,
        prompt_masks: torch.Tensor = None,
        multimask_output: bool = True,
    ):
        """Predict masks given point prompts.

        Args:
            coords: [B, N, 3]. Point cloud coordinates, normalized to [-1, 1].
            features: [B, N, F]. Point cloud features.
        """
        # pc_embeddings: [B, num_patches, D]
        pc_embeddings, patches = self.pc_encoder(coords, features)
        centers = patches["centers"]  # [B, num_patches, 3]
        knn_idx = patches["knn_idx"]  # [B, N, K]
        aux_inputs = AuxInputs(coords=coords, features=features, centers=centers)

        # [B, num_patches, D]
        pc_pe = self.point_encoder.pe_layer(centers)

        # [B * M, num_queries, D]
        sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)
        
        # [B * M, num_patches, D] or [B, num_patches, D] (if prompt_masks=None)
        dense_embeddings = self.mask_encoder(
            prompt_masks,
            coords,
            centers,
            knn_idx
        )

        # [B * M, num_patches, D]
        dense_embeddings = repeat_interleave(
            dense_embeddings,
            sparse_embeddings.shape[0] // dense_embeddings.shape[0],
            0,
        )
        
        # [B * M, num_outputs, N], [B * M, num_outputs]
        masks, iou_preds = self.mask_decoder(
            pc_embeddings,
            pc_pe,
            sparse_embeddings,
            dense_embeddings,
            aux_inputs=aux_inputs,
            multimask_output=multimask_output,
        )
        return masks, iou_preds

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        gt_masks: torch.Tensor,
        is_eval: torch.bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """Forward pass for training. The prompts are sampled given the ground truth masks.

        Args:
            coords: [B, N, 3]. Point cloud coordinates, normalized to [-1, 1].
            features: [B, N, F]. Point cloud features.
            gt_masks: [B, M, N], bool. Ground truth binary masks.

        Returns:
            outputs: List of dictionaries. Each dictionary contains the following keys:
                - prompt_coords: [B * M, num_queries, 3]. Coordinates of the sampled prompts.
                - prompt_labels: [B * M, num_queries], bool. Labels of the sampled prompts.
                - prompt_masks: [B * M, N]. The most confident mask.
                - masks: [B * M, num_outputs, N]. Predicted masks.
                - iou_preds: [B * M, num_outputs]. IoU predictions.
        """
        batch_size = coords.shape[0]
        num_masks = gt_masks.shape[1]

        # pc_embeddings: [B, num_patches, D]
        pc_embeddings, patches = self.pc_encoder(coords, features)
        centers = patches["centers"]  # [B, num_patches, 3]
        knn_idx = patches["knn_idx"]  # [B, N, K]

        outputs = []  # Store the output at each iteration
        prompt_coords = coords.new_empty((batch_size * num_masks, 0, 3))
        prompt_labels = gt_masks.new_empty((batch_size * num_masks, 0))
        prompt_masks = None  # [B * M, N]
        aux_inputs = AuxInputs(coords=coords, features=features, centers=centers)

        # According to Appendix A (training algorithm) of SAM paper,
        # there are two iterations where no additional prompts are sampled.
        if self.enable_mask_refinement_iterations and self.training:
            mask_refinement_iterations = [self.prompt_iters - 1]
            if self.prompt_iters > 1:
                sampled_iter = torch.randint(1, self.prompt_iters, (1,)).item()
                mask_refinement_iterations.append(sampled_iter)
        else:
            mask_refinement_iterations = []

        # [B, num_patches, D]
        pc_pe = self.point_encoder.pe_layer(centers)

        for i in range(self.prompt_iters):
            if i == 0 or i not in mask_refinement_iterations:
                new_prompt_coords, new_prompt_labels = sample_prompts_adapter(
                    coords, gt_masks, prompt_masks, is_eval=is_eval,
                )
                prompt_coords = torch.cat([prompt_coords, new_prompt_coords], dim=1)
                prompt_labels = torch.cat([prompt_labels, new_prompt_labels], dim=1)

            # [B * M, num_queries, D]
            sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)

            # [B * M, num_patches, D] or [B, num_patches, D] (if prompt_masks=None)
            dense_embeddings = self.mask_encoder(
                prompt_masks,
                coords,
                centers,
                knn_idx,
                center_idx=patches.get("fps_idx"),
            )
            # [B * M, num_patches, D]
            dense_embeddings = repeat_interleave(
                dense_embeddings,
                sparse_embeddings.shape[0] // dense_embeddings.shape[0],
                0,
            )

            # [B * M, num_outputs, N], [B * M, num_outputs]
            masks, iou_preds = self.mask_decoder(
                pc_embeddings,
                pc_pe,
                sparse_embeddings,
                dense_embeddings,
                aux_inputs=aux_inputs,
                multimask_output=(i == 0),
            )

            # Select the most confident mask for the next iteration
            if i == 0:
                max_iou_pred_ind = torch.argmax(iou_preds, dim=1)  # [B * M]
                prompt_masks = batch_index_select(
                    masks, max_iou_pred_ind, dim=1
                )  # [B * M, N]
            else:
                max_iou_pred_ind = 0
                prompt_masks = masks[:, 0]

            outputs.append(
                dict(
                    prompt_coords=prompt_coords,
                    prompt_labels=prompt_labels,
                    masks=masks,
                    iou_preds=iou_preds,
                    max_iou_pred_ind=max_iou_pred_ind,
                    prompt_masks=prompt_masks,
                )
            )

        return outputs


class PointCloudSAMNN(nn.Module):
    def __init__(
        self,
        pc_encoder: PointCloudEncoder,
        mask_encoder: MaskEncoder,
        mask_decoder: MaskDecoder,
        prompt_iters: int,
        enable_mask_refinement_iterations=True,
    ):
        super().__init__()
        self.pc_encoder = pc_encoder
        self.point_encoder = PointEncoder(pc_encoder.embed_dim)
        self.mask_encoder = mask_encoder
        self.mask_decoder = mask_decoder
        self.prompt_iters = prompt_iters
        self.enable_mask_refinement_iterations = enable_mask_refinement_iterations

    def predict_masks(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        prompt_coords: torch.Tensor,
        prompt_labels: torch.Tensor,
        prompt_masks: torch.Tensor = None,
        multimask_output: bool = True,
    ):
        """Predict masks given point prompts.

        Args:
            coords: [B, N, 3]. Point cloud coordinates, normalized to [-1, 1].
            features: [B, N, F]. Point cloud features.
        """
        # pc_embeddings: [B, num_patches, D]
        pc_embeddings, patches = self.pc_encoder(coords, features)
        centers = patches["centers"]  # [B, num_patches, 3]
        knn_idx = patches["knn_idx"]  # [B, N, K]
        aux_inputs = AuxInputs(coords=coords, features=features, centers=centers)

        # [B, num_patches, D]
        pc_pe = self.point_encoder.pe_layer(centers)

        # [B * M, num_queries, D]
        sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)
        
        # [B * M, num_patches, D] or [B, num_patches, D] (if prompt_masks=None)
        dense_embeddings = self.mask_encoder(
            prompt_masks,
            coords,
            centers,
            knn_idx
        )

        # [B * M, num_patches, D]
        dense_embeddings = repeat_interleave(
            dense_embeddings,
            sparse_embeddings.shape[0] // dense_embeddings.shape[0],
            0,
        )
        
        # [B * M, num_outputs, N], [B * M, num_outputs]
        masks, iou_preds = self.mask_decoder(
            pc_embeddings,
            pc_pe,
            sparse_embeddings,
            dense_embeddings,
            aux_inputs=aux_inputs,
            multimask_output=multimask_output,
        )
        return masks, iou_preds

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        gt_masks: torch.Tensor,
        is_eval: torch.bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """Forward pass for training. The prompts are sampled given the ground truth masks.

        Args:
            coords: [B, N, 3]. Point cloud coordinates, normalized to [-1, 1].
            features: [B, N, F]. Point cloud features.
            gt_masks: [B, M, N], bool. Ground truth binary masks.

        Returns:
            outputs: List of dictionaries. Each dictionary contains the following keys:
                - prompt_coords: [B * M, num_queries, 3]. Coordinates of the sampled prompts.
                - prompt_labels: [B * M, num_queries], bool. Labels of the sampled prompts.
                - prompt_masks: [B * M, N]. The most confident mask.
                - masks: [B * M, num_outputs, N]. Predicted masks.
                - iou_preds: [B * M, num_outputs]. IoU predictions.
        """
        batch_size = coords.shape[0]
        num_masks = gt_masks.shape[1]

        # pc_embeddings: [B, num_patches, D]
        pc_embeddings, patches = self.pc_encoder(coords, features)
        centers = patches["centers"]  # [B, num_patches, 3]
        nn_idx = patches["nn_idx"]  # [B, N, K]

        outputs = []  # Store the output at each iteration
        prompt_coords = coords.new_empty((batch_size * num_masks, 0, 3))
        prompt_labels = gt_masks.new_empty((batch_size * num_masks, 0))
        prompt_masks = None  # [B * M, N]
        aux_inputs = AuxInputs(coords=coords, features=features, centers=centers)

        # According to Appendix A (training algorithm) of SAM paper,
        # there are two iterations where no additional prompts are sampled.
        if self.enable_mask_refinement_iterations and self.training:
            mask_refinement_iterations = [self.prompt_iters - 1]
            if self.prompt_iters > 1:
                sampled_iter = torch.randint(1, self.prompt_iters, (1,)).item()
                mask_refinement_iterations.append(sampled_iter)
        else:
            mask_refinement_iterations = []

        # [B, num_patches, D]
        pc_pe = self.point_encoder.pe_layer(centers)

        for i in range(self.prompt_iters):
            if i == 0 or i not in mask_refinement_iterations:
                new_prompt_coords, new_prompt_labels = sample_prompts_adapter(
                    coords, gt_masks, prompt_masks, is_eval=is_eval,
                )
                prompt_coords = torch.cat([prompt_coords, new_prompt_coords], dim=1)
                prompt_labels = torch.cat([prompt_labels, new_prompt_labels], dim=1)

            # [B * M, num_queries, D]
            sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)

            # [B * M, num_patches, D] or [B, num_patches, D] (if prompt_masks=None)
            dense_embeddings = self.mask_encoder(
                prompt_masks,
                nn_idx,
                centers,
                coords,
            )
            # [B * M, num_patches, D]
            dense_embeddings = repeat_interleave(
                dense_embeddings,
                sparse_embeddings.shape[0] // dense_embeddings.shape[0],
                0,
            )

            # [B * M, num_outputs, N], [B * M, num_outputs]
            masks, iou_preds = self.mask_decoder(
                pc_embeddings,
                pc_pe,
                sparse_embeddings,
                dense_embeddings,
                aux_inputs=aux_inputs,
                multimask_output=(i == 0),
            )

            # Select the most confident mask for the next iteration
            if i == 0:
                max_iou_pred_ind = torch.argmax(iou_preds, dim=1)  # [B * M]
                prompt_masks = batch_index_select(
                    masks, max_iou_pred_ind, dim=1
                )  # [B * M, N]
            else:
                max_iou_pred_ind = 0
                prompt_masks = masks[:, 0]

            outputs.append(
                dict(
                    prompt_coords=prompt_coords,
                    prompt_labels=prompt_labels,
                    masks=masks,
                    iou_preds=iou_preds,
                    max_iou_pred_ind=max_iou_pred_ind,
                    prompt_masks=prompt_masks,
                )
            )

        return outputs


class PointCloudSAMHier(PointCloudSAM):
    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Forward pass for training. The prompts are sampled given the ground truth masks.

        Args:
            coords: [B, N, 3]. Point cloud coordinates, normalized to [-1, 1].
            features: [B, N, F]. Point cloud features.
            gt_masks: [B, M, N], bool. Ground truth binary masks.

        Returns:
            outputs: List of dictionaries. Each dictionary contains the following keys:
                - prompt_coords: [B * M, num_queries, 3]. Coordinates of the sampled prompts.
                - prompt_labels: [B * M, num_queries], bool. Labels of the sampled prompts.
                - prompt_masks: [B * M, N]. The most confident mask.
                - masks: [B * M, num_outputs, N]. Predicted masks.
                - iou_preds: [B * M, num_outputs]. IoU predictions.
        """
        batch_size = coords.shape[0]
        num_masks = gt_masks.shape[1]

        # pc_embeddings, [B, num_patches, D]
        pc_embeddings, patches = self.pc_encoder(coords, features)
        centers = patches[-1]["centers"]  # [B, num_patches, 3]

        outputs = []  # Store the output at each iteration
        prompt_coords = coords.new_empty((batch_size * num_masks, 0, 3))
        prompt_labels = gt_masks.new_empty((batch_size * num_masks, 0))
        prompt_masks = None  # [B * M, N]
        aux_inputs1 = AuxInputs(
            coords=coords, features=features, centers=patches[0]["centers"]
        )
        aux_inputs2 = AuxInputs(
            coords=patches[0]["centers"],
            features=patches[0]["embeddings"],
            centers=patches[1]["centers"],
        )

        # According to Appendix A (training algorithm) of SAM paper,
        # there are two iterations where no additional prompts are sampled.
        if self.enable_mask_refinement_iterations and self.training:
            mask_refinement_iterations = [self.prompt_iters - 1]
            if self.prompt_iters > 1:
                sampled_iter = torch.randint(1, self.prompt_iters, (1,)).item()
                mask_refinement_iterations.append(sampled_iter)
        else:
            mask_refinement_iterations = []

        # [B, num_patches, D]
        pc_pe = self.point_encoder.pe_layer(centers)

        for i in range(self.prompt_iters):
            if i == 0 or i not in mask_refinement_iterations:
                new_prompt_coords, new_prompt_labels = sample_prompts(
                    coords, gt_masks, prompt_masks
                )
                prompt_coords = torch.cat([prompt_coords, new_prompt_coords], dim=1)
                prompt_labels = torch.cat([prompt_labels, new_prompt_labels], dim=1)

            # [B * M, num_queries, D]
            sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)

            # [B * M, num_patches, D] or [B, num_patches, D] (if prompt_masks=None)
            dense_embeddings = self.mask_encoder(
                prompt_masks,
                coords,
                patches[0]["centers"],
                patches[0]["knn_idx"],
                patches[1]["centers"],
                patches[1]["knn_idx"],
            )
            if isinstance(dense_embeddings, torch.Tensor):
                # [B * M, num_patches, D]
                dense_embeddings = repeat_interleave(
                    dense_embeddings,
                    sparse_embeddings.shape[0] // dense_embeddings.shape[0],
                    0,
                )
            elif isinstance(dense_embeddings, list):
                dense_embeddings = dense_embeddings[-1]
            else:
                raise ValueError(type(dense_embeddings))

            # [B * M, num_outputs, N], [B * M, num_outputs]
            masks, iou_preds = self.mask_decoder(
                pc_embeddings,
                pc_pe,
                sparse_embeddings,
                dense_embeddings,
                aux_inputs1=aux_inputs1,
                aux_inputs2=aux_inputs2,
                multimask_output=(i == 0),
            )

            # Select the most confident mask for the next iteration
            if i == 0:
                max_iou_pred_ind = torch.argmax(iou_preds, dim=1)  # [B * M]
                prompt_masks = batch_index_select(
                    masks, max_iou_pred_ind, dim=1
                )  # [B * M, N]
            else:
                max_iou_pred_ind = 0
                prompt_masks = masks[:, 0]

            outputs.append(
                dict(
                    prompt_coords=prompt_coords,
                    prompt_labels=prompt_labels,
                    masks=masks,
                    iou_preds=iou_preds,
                    max_iou_pred_ind=max_iou_pred_ind,
                    prompt_masks=prompt_masks,
                )
            )

        return outputs


def main():
    import timm

    from .pc_encoder import PatchEmbed
    from .transformer import TwoWayTransformer

    model_name = "vit_base_patch16_224"
    embed_dim = 256
    transformer_encoder = timm.create_model(model_name, pretrained=False)
    patch_embed = PatchEmbed(6, embed_dim, 512, 64)
    pc_encoder = PointCloudEncoder(patch_embed, transformer_encoder, embed_dim)
    mask_encoder = MaskEncoder(embed_dim)
    transformer_decoder = TwoWayTransformer(2, embed_dim, 8, 2048)
    mask_decoder = MaskDecoder(embed_dim, transformer_decoder)
    model = PointCloudSAM(pc_encoder, mask_encoder, mask_decoder, 3).cuda()
    print("Model created.")

    points = torch.rand(2, 1024, 3).cuda() * 2 - 1
    point_colors = torch.rand(2, 1024, 3).cuda()
    gt_masks = torch.randint(0, 2, [2, 2, 1024]).bool().cuda()
    with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
        outputs = model(points, point_colors, gt_masks)

    for i, output in enumerate(outputs):
        print(f"Iteration {i}")
        print(output["prompt_coords"].shape)
        print(output["prompt_labels"].shape)
        print(output["masks"].shape)
        print(output["iou_preds"].shape)


def main_hier():
    import timm

    from .mask_decoder import MaskDecoderHier
    from .pc_encoder import PatchEmbedHier
    from .prompt_encoder import MaskEncoderHier
    from .transformer import TwoWayTransformer

    model_name = "vit_base_patch16_224"
    embed_dim = 256
    transformer_encoder = timm.create_model(model_name, pretrained=False)
    patch_embed = PatchEmbedHier(6, embed_dim, [2048, 512], [32, 32])
    pc_encoder = PointCloudEncoder(patch_embed, transformer_encoder, embed_dim)
    mask_encoder = MaskEncoderHier(embed_dim)
    transformer_decoder = TwoWayTransformer(2, embed_dim, 8, 2048)
    mask_decoder = MaskDecoderHier(embed_dim, transformer_decoder)
    model = PointCloudSAMHier(pc_encoder, mask_encoder, mask_decoder, 3).cuda()
    print("Model created.")

    n = 4096
    points = torch.rand(2, n, 3).cuda() * 2 - 1
    point_colors = torch.rand(2, n, 3).cuda()
    gt_masks = torch.randint(0, 2, [2, 2, n]).bool().cuda()
    with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
        outputs = model(points, point_colors, gt_masks)

    for i, output in enumerate(outputs):
        print(f"Iteration {i}")
        print(output["prompt_coords"].shape)
        print(output["prompt_labels"].shape)
        print(output["masks"].shape)
        print(output["iou_preds"].shape)


if __name__ == "__main__":
    # main()
    main_hier()
