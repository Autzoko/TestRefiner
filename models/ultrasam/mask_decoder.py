"""SAM Mask Decoder — standalone PyTorch, no mmdet/mmcv/mmengine.

Based on the original SAMHead mask prediction logic.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn

from .common import MLP, LayerNorm2d


class SAMMaskDecoder(nn.Module):
    """Decodes transformer output tokens into predicted masks and IoU scores.

    Expected input flow:
        prompt_encoder  -->  transformer  -->  mask_decoder
    """

    def __init__(
        self,
        transformer_dim: int = 256,
        num_multimask_outputs: int = 3,
        activation: type = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1  # 4

        # Upscaling: (B, 256, 64, 64) -> (B, 32, 256, 256)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # One MLP per mask token: maps (transformer_dim) -> (transformer_dim // 8)
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, num_layers=3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        # IoU prediction head
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth,
        )

    def forward(
        self,
        point_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        prompt_padding_masks: torch.Tensor,
        multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict masks and IoU scores.

        Args:
            point_embedding:      (B, N_tokens, C) — output of the transformer
                                  decoder (queries). The last 5 tokens are
                                  [MASK_OUT, MASK_OUT_1, MASK_OUT_2, MASK_OUT_3, IOU_OUT].
            image_embedding:      (B, HW, C) — refined image keys from the
                                  transformer decoder.
            prompt_padding_masks: (B, N_tokens) — 0 = real prompt, 1 = padding
            multimask_output:     if True return 3 masks (indices 1-3),
                                  else return 1 mask (index 0).

        Returns:
            masks:    (B, n_masks, 256, 256) — predicted mask logits
            iou_pred: (B, n_masks)           — predicted IoU scores
        """
        B = point_embedding.shape[0]
        C = self.transformer_dim

        # ---- Identify active (non-fully-padded) instances ------------
        # A prompt is "active" if at least one of its non-output tokens
        # is not padding.  Output tokens are never counted as padding.
        n_output = self.num_mask_tokens + 1  # 5 (4 mask + 1 iou)
        n_prompt_tokens = point_embedding.shape[1] - n_output
        if n_prompt_tokens > 0:
            prompt_part = prompt_padding_masks[:, :n_prompt_tokens]  # (B, n_prompt)
            active_mask = (prompt_part.sum(dim=1) < n_prompt_tokens)  # (B,)
        else:
            active_mask = torch.ones(B, dtype=torch.bool, device=point_embedding.device)

        # ---- Extract mask tokens and IoU token -----------------------
        # Layout: [...prompt..., MASK_OUT, MASK_OUT_1, MASK_OUT_2, MASK_OUT_3, IOU_OUT]
        mask_tokens = point_embedding[:, -(n_output):-1, :]  # (B, 4, C)
        iou_token = point_embedding[:, -1, :]                # (B, C)

        # ---- Reshape image embedding to spatial ----------------------
        # image_embedding: (B, HW, C) -> (B, C, H, W) where H = W = sqrt(HW)
        HW = image_embedding.shape[1]
        H = W = int(math.isqrt(HW))
        assert H * W == HW, f"image_embedding spatial dim {HW} is not a perfect square"
        image_feat = image_embedding.permute(0, 2, 1).reshape(B, C, H, W)

        # ---- Upscale image features ----------------------------------
        upscaled = self.output_upscaling(image_feat)  # (B, C//8, 4H, 4W)
        up_C = upscaled.shape[1]
        up_H = upscaled.shape[2]
        up_W = upscaled.shape[3]

        # ---- Hypernetwork: mask_tokens -> pixel-level weights --------
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens[:, i, :])
            )  # each (B, C//8)
        hyper_in = torch.stack(hyper_in_list, dim=1)  # (B, 4, C//8)

        # ---- Predicted masks -----------------------------------------
        # (B, 4, C//8) @ (B, C//8, up_H*up_W) -> (B, 4, up_H*up_W)
        masks = hyper_in @ upscaled.reshape(B, up_C, up_H * up_W)
        masks = masks.reshape(B, self.num_mask_tokens, up_H, up_W)  # (B, 4, 256, 256)

        # ---- IoU prediction ------------------------------------------
        iou_pred = self.iou_prediction_head(iou_token)  # (B, 4)

        # ---- Select output based on multimask flag -------------------
        if multimask_output:
            masks = masks[:, 1:, :, :]      # (B, 3, 256, 256)
            iou_pred = iou_pred[:, 1:]       # (B, 3)
        else:
            masks = masks[:, 0:1, :, :]      # (B, 1, 256, 256)
            iou_pred = iou_pred[:, 0:1]      # (B, 1)

        # Zero out predictions for fully-padded (inactive) instances
        if not active_mask.all():
            inactive = ~active_mask
            masks[inactive] = 0.0
            iou_pred[inactive] = 0.0

        return masks, iou_pred
