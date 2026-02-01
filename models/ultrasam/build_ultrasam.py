"""Top-level UltraSAM model combining all sub-modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .image_encoder import ImageEncoderViT
from .prompt_encoder import SAMPromptEncoder, N_OUTPUT_TOKENS, N_TOTAL_TOKENS
from .transformer import SAMTransformerDecoder
from .mask_decoder import SAMMaskDecoder


class UltraSAM(nn.Module):
    """UltraSAM: SAM-based model for ultrasound image segmentation.

    Combines:
        - ImageEncoderViT: ViT-B backbone producing (B, 256, 64, 64) features
        - SAMPromptEncoder: encodes point/box prompts into token embeddings
        - SAMTransformerDecoder: two-way cross-attention between prompts and image
        - SAMMaskDecoder: predicts masks (256x256) and IoU scores from decoder output
    """

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: SAMPromptEncoder,
        transformer_decoder: SAMTransformerDecoder,
        mask_decoder: SAMMaskDecoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.transformer_decoder = transformer_decoder
        self.mask_decoder = mask_decoder

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images through the ViT backbone.

        Args:
            images: (B, 3, 1024, 1024) normalized images
        Returns:
            (B, 256, 64, 64) image embeddings
        """
        return self.image_encoder(images)

    def forward(
        self,
        images: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        box_labels: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            images:       (B, 3, 1024, 1024) — normalized input images
            points:       (B, N, 2) — point coordinates in 1024x1024 space
            point_labels: (B, N) — integer labels per point
            boxes:        (B, N_boxes, 2, 2) — box corner coordinates
            box_labels:   (B, N_boxes, 2) — integer labels per box corner
            multimask_output: if True return 3 masks, else 1

        Returns:
            masks:    (B, n_masks, 256, 256) — mask logits
            iou_pred: (B, n_masks) — predicted IoU scores
        """
        # 1. Encode image
        image_embedding = self.image_encoder(images)  # (B, 256, 64, 64)

        # 2. Encode prompts
        prompt_out = self.prompt_encoder.encode_points_and_boxes(
            points=points,
            point_labels=point_labels,
            boxes=boxes,
            box_labels=box_labels,
        )
        pts_embed = prompt_out["pts_embed"]               # (B, 7, 256)
        dense_embed = prompt_out["dense_embed"]            # (B, 256, 64, 64)
        prompt_padding_masks = prompt_out["prompt_padding_masks"]  # (B, 7)

        # 3. Add dense prompt embedding to image embedding
        image_with_dense = image_embedding + dense_embed   # (B, 256, 64, 64)

        # 4. Get positional encoding for image
        image_pe = self.prompt_encoder.get_dense_pe()      # (1, 256, 64, 64)

        # 5. Run transformer decoder
        queries, keys = self.transformer_decoder(
            image_embedding=image_with_dense,
            image_pos=image_pe,
            query_pos=pts_embed,
        )  # queries: (B, 7, 256), keys: (B, 4096, 256)

        # 6. Decode masks
        masks, iou_pred = self.mask_decoder(
            point_embedding=queries,
            image_embedding=keys,
            prompt_padding_masks=prompt_padding_masks,
            multimask_output=multimask_output,
        )

        return masks, iou_pred

    def forward_with_prompts(
        self,
        images: torch.Tensor,
        image_embeddings: Optional[torch.Tensor] = None,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        box_coords: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified interface for the pipeline.

        Accepts point coords (B, N, 2) and box coords (B, 4) as simple tensors.
        Box coords are (x1, y1, x2, y2) and get converted to the (B, 1, 2, 2) format.

        Args:
            images:           (B, 3, 1024, 1024) — can be None if image_embeddings provided
            image_embeddings: (B, 256, 64, 64) — precomputed, skip encoder if provided
            point_coords:     (B, N, 2) — point coordinates
            point_labels:     (B, N) — point labels (use 2 for positive)
            box_coords:       (B, 4) — box as (x1, y1, x2, y2)
            multimask_output: if True return 3 masks, else 1

        Returns:
            masks:    (B, n_masks, 256, 256) — mask logits
            iou_pred: (B, n_masks) — predicted IoU scores
        """
        # Encode image if embeddings not provided
        if image_embeddings is None:
            assert images is not None, "Either images or image_embeddings must be provided"
            image_embeddings = self.image_encoder(images)

        # Convert box_coords (B, 4) to (B, 1, 2, 2) format
        boxes = None
        box_labels = None
        if box_coords is not None:
            B = box_coords.shape[0]
            boxes = box_coords.reshape(B, 1, 2, 2)  # (x1,y1) and (x2,y2) corners
            box_labels = torch.tensor([[3, 4]], dtype=torch.long,
                                       device=box_coords.device).expand(B, -1)  # BOX_CORNER_A, BOX_CORNER_B

        # Format point labels
        points = point_coords  # (B, N, 2) or None

        # Encode prompts
        prompt_out = self.prompt_encoder.encode_points_and_boxes(
            points=points,
            point_labels=point_labels,
            boxes=boxes,
            box_labels=box_labels,
        )
        pts_embed = prompt_out["pts_embed"]
        dense_embed = prompt_out["dense_embed"]
        prompt_padding_masks = prompt_out["prompt_padding_masks"]

        # Add dense embed to image
        image_with_dense = image_embeddings + dense_embed

        # Image positional encoding
        image_pe = self.prompt_encoder.get_dense_pe()

        # Transformer decoder
        queries, keys = self.transformer_decoder(
            image_embedding=image_with_dense,
            image_pos=image_pe,
            query_pos=pts_embed,
        )

        # Mask decoder
        masks, iou_pred = self.mask_decoder(
            point_embedding=queries,
            image_embedding=keys,
            prompt_padding_masks=prompt_padding_masks,
            multimask_output=multimask_output,
        )

        return masks, iou_pred


def build_ultrasam_vit_b(
    img_size: int = 1024,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    patch_size: int = 16,
    out_chans: int = 256,
    window_size: int = 14,
    global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),
) -> UltraSAM:
    """Build UltraSAM with ViT-B image encoder (default config)."""

    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        out_chans=out_chans,
        window_size=window_size,
        global_attn_indexes=list(global_attn_indexes),
    )

    prompt_encoder = SAMPromptEncoder(
        embed_dim=out_chans,
        image_embedding_size=(img_size // patch_size, img_size // patch_size),
        input_image_size=(img_size, img_size),
    )

    transformer_decoder = SAMTransformerDecoder(
        num_layers=2,
        num_heads=8,
        embedding_dim=out_chans,
        mlp_dim=2048,
        attention_downsample_rate=2,
    )

    mask_decoder = SAMMaskDecoder(
        transformer_dim=out_chans,
        num_multimask_outputs=3,
    )

    return UltraSAM(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        transformer_decoder=transformer_decoder,
        mask_decoder=mask_decoder,
    )
