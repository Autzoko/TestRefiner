"""SAM Prompt Encoder — standalone PyTorch, no mmdet/mmcv/mmengine.

Based on the original UltraSam SAMPaddingGenerator.
"""

from enum import Enum, auto
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn

from .common import PositionEmbeddingRandom, LayerNorm2d


class EmbeddingIndex(Enum):
    """Indices into the label embedding table."""
    NON_INIT_MASK_EMBED = 0
    NEG = auto()            # 1
    POS = auto()            # 2
    BOX_CORNER_A = auto()   # 3
    BOX_CORNER_B = auto()   # 4
    NOT_A_POINT = auto()    # 5
    MASK_OUT = auto()       # 6
    MASK_OUT_1 = auto()     # 7
    MASK_OUT_2 = auto()     # 8
    MASK_OUT_3 = auto()     # 9
    IOU_OUT = auto()        # 10


# Number of output tokens appended to every prompt
_OUTPUT_TOKEN_LABELS = [
    EmbeddingIndex.MASK_OUT,
    EmbeddingIndex.MASK_OUT_1,
    EmbeddingIndex.MASK_OUT_2,
    EmbeddingIndex.MASK_OUT_3,
    EmbeddingIndex.IOU_OUT,
]
N_OUTPUT_TOKENS = len(_OUTPUT_TOKEN_LABELS)  # 5
N_PROMPT_SLOTS = 2  # always pad prompt to 2 slots
N_TOTAL_TOKENS = N_PROMPT_SLOTS + N_OUTPUT_TOKENS  # 7


class SAMPromptEncoder(nn.Module):
    """Encodes point / box / mask prompts into the token+dense format
    expected by the SAM transformer decoder.

    This is a *simplified* version of the original UltraSam
    ``SAMPaddingGenerator`` that works on plain tensors instead of
    mmdet DataSamples.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        input_image_size: Tuple[int, int] = (1024, 1024),
        mask_in_chans: int = 16,
        n_output_tokens: int = N_OUTPUT_TOKENS,
        use_mask_refinement: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.n_output_tokens = n_output_tokens

        # Positional encoding layer
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Label embedding: one vector per EmbeddingIndex entry
        self.label_encoder = nn.Embedding(len(EmbeddingIndex), embed_dim)

        # Optional mask refinement branch
        self.use_mask_refinement = use_mask_refinement
        if use_mask_refinement:
            self.mask_downscaling = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=2, stride=2),
                LayerNorm2d(4),
                nn.GELU(),
                nn.Conv2d(4, mask_in_chans, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans),
                nn.GELU(),
                nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
            )

    # ------------------------------------------------------------------
    # Positional encoding helpers
    # ------------------------------------------------------------------

    def get_dense_pe(self) -> torch.Tensor:
        """Return positional encoding for the image grid: (1, C, H, W)."""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        coords: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute point embeddings = label_embed + positional_embed.

        Args:
            coords: (B, N, 2)  in pixel space (0..input_image_size)
            labels: (B, N)     integer labels (EmbeddingIndex values)
        Returns:
            (B, N, embed_dim)
        """
        pos_embed = self.pe_layer.forward_with_coords(
            coords, self.input_image_size,
        )  # (B, N, embed_dim)
        label_embed = self.label_encoder(labels)  # (B, N, embed_dim)
        return pos_embed + label_embed

    # ------------------------------------------------------------------
    # Main encoding entry-point
    # ------------------------------------------------------------------

    def encode_points_and_boxes(
        self,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        box_labels: Optional[torch.Tensor] = None,
        mask_props: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode sparse (point / box) and dense (mask) prompts.

        Args:
            points:       (B, N_points, 2) — coordinates in image space
            point_labels: (B, N_points)    — per-point label integers
            boxes:        (B, N_boxes, 2, 2) — two corner coords per box
            box_labels:   (B, N_boxes, 2)    — label per corner
            mask_props:   (B, 1, 256, 256)   — mask proposal (optional)

        Returns a dict:
            pts_embed            (B, N_total, embed_dim)
            dense_embed          (B, embed_dim, H_emb, W_emb)
            prompt_padding_masks (B, N_total)   0=real, 1=padding
            padded_points        (B, N_total, 2)
            padded_labels        (B, N_total)
        """
        has_points = points is not None and point_labels is not None
        has_boxes = boxes is not None and box_labels is not None
        assert has_points or has_boxes, (
            "At least one of (points, point_labels) or (boxes, box_labels) must be provided."
        )

        device = points.device if has_points else boxes.device
        B = points.shape[0] if has_points else boxes.shape[0]

        # --- Build the 2 prompt-slot coords and labels ----------------
        # Slot layout: [slot0, slot1]
        # Then appended: [MASK_OUT, MASK_OUT_1, MASK_OUT_2, MASK_OUT_3, IOU_OUT]

        padded_coords = torch.zeros(B, N_PROMPT_SLOTS, 2, device=device)
        padded_labels = torch.full(
            (B, N_PROMPT_SLOTS), EmbeddingIndex.NOT_A_POINT.value,
            dtype=torch.long, device=device,
        )
        padding_mask = torch.ones(B, N_PROMPT_SLOTS, device=device)  # 1 = padding

        if has_points and not has_boxes:
            # slot0 = first point, slot1 = NOT_A_POINT padding
            padded_coords[:, 0, :] = points[:, 0, :]
            padded_labels[:, 0] = point_labels[:, 0]
            padding_mask[:, 0] = 0.0
            # slot1 stays NOT_A_POINT (padding)
            padded_coords[:, 1, :] = 0.0

        elif has_boxes and not has_points:
            # slot0 = BOX_CORNER_A, slot1 = BOX_CORNER_B
            padded_coords[:, 0, :] = boxes[:, 0, 0, :]  # first box, corner A
            padded_coords[:, 1, :] = boxes[:, 0, 1, :]  # first box, corner B
            padded_labels[:, 0] = box_labels[:, 0, 0]
            padded_labels[:, 1] = box_labels[:, 0, 1]
            padding_mask[:, 0] = 0.0
            padding_mask[:, 1] = 0.0

        else:
            # Both points and boxes: slot0 = point, slot1 = BOX_CORNER_A
            # (simple strategy — use first point and first box corner A)
            padded_coords[:, 0, :] = points[:, 0, :]
            padded_labels[:, 0] = point_labels[:, 0]
            padded_coords[:, 1, :] = boxes[:, 0, 0, :]
            padded_labels[:, 1] = box_labels[:, 0, 0]
            padding_mask[:, 0] = 0.0
            padding_mask[:, 1] = 0.0

        # --- Append output tokens ------------------------------------
        output_token_labels = torch.tensor(
            [e.value for e in _OUTPUT_TOKEN_LABELS],
            dtype=torch.long, device=device,
        ).unsqueeze(0).expand(B, -1)  # (B, 5)

        # Output tokens get zero coordinates (PE will be zeroed out below)
        output_token_coords = torch.zeros(B, self.n_output_tokens, 2, device=device)
        output_padding_mask = torch.zeros(B, self.n_output_tokens, device=device)  # not padding

        all_coords = torch.cat([padded_coords, output_token_coords], dim=1)  # (B, 7, 2)
        all_labels = torch.cat([padded_labels, output_token_labels], dim=1)   # (B, 7)
        all_padding = torch.cat([padding_mask, output_padding_mask], dim=1)   # (B, 7)

        # --- Compute sparse (point) embeddings -----------------------
        # Label embeddings
        label_embed = self.label_encoder(all_labels)  # (B, 7, C)

        # Positional embeddings (for prompt slots only; zero for output tokens)
        pos_embed = self.pe_layer.forward_with_coords(
            all_coords, self.input_image_size,
        )  # (B, 7, C)
        # Zero out PE for the output tokens
        pos_embed[:, N_PROMPT_SLOTS:, :] = 0.0

        pts_embed = label_embed + pos_embed  # (B, 7, C)

        # --- Compute dense embeddings --------------------------------
        H_emb, W_emb = self.image_embedding_size
        if mask_props is not None and self.use_mask_refinement:
            dense_embed = self.mask_downscaling(mask_props)  # (B, C, H_emb, W_emb)
        else:
            # Use the NON_INIT_MASK_EMBED as a broadcast dense embedding
            no_mask_embed = self.label_encoder.weight[
                EmbeddingIndex.NON_INIT_MASK_EMBED.value
            ]  # (C,)
            dense_embed = no_mask_embed.reshape(1, self.embed_dim, 1, 1).expand(
                B, -1, H_emb, W_emb,
            )

        return {
            "pts_embed": pts_embed,                  # (B, 7, C)
            "dense_embed": dense_embed,              # (B, C, 64, 64)
            "prompt_padding_masks": all_padding,     # (B, 7)
            "padded_points": all_coords,             # (B, 7, 2)
            "padded_labels": all_labels,             # (B, 7)
        }
