"""Visualization utilities for comparing segmentation results."""

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _draw_prompts(ax: plt.Axes, prompts: Dict) -> None:
    """Draw box and point prompts on a matplotlib axis."""
    bbox = prompts.get("box")
    point = prompts.get("point")
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="yellow", facecolor="none",
        )
        ax.add_patch(rect)
    if point is not None:
        ax.plot(point[0], point[1], "r*", markersize=15)


def visualize_comparison(
    image: np.ndarray,
    gt: np.ndarray,
    transunet_pred: np.ndarray,
    ultrasam_pred: np.ndarray,
    prompts: Optional[Dict] = None,
    save_path: Optional[str] = None,
) -> None:
    """Create 4-panel comparison figure.

    Panels: (1) Image + GT overlay, (2) TransUNet pred + prompts,
    (3) UltraSAM pred + prompts, (4) Image + prompts only.

    Box prompt is drawn as a yellow rectangle, point prompt as a red star.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Image + GT overlay
    axes[0].imshow(image)
    _overlay_mask(axes[0], gt, color="green", alpha=0.35)
    axes[0].set_title("Image + GT")
    axes[0].axis("off")

    # Panel 2: TransUNet prediction + prompts
    axes[1].imshow(image)
    _overlay_mask(axes[1], transunet_pred, color="blue", alpha=0.35)
    if prompts is not None:
        _draw_prompts(axes[1], prompts)
    axes[1].set_title("TransUNet")
    axes[1].axis("off")

    # Panel 3: UltraSAM prediction + prompts
    axes[2].imshow(image)
    _overlay_mask(axes[2], ultrasam_pred, color="red", alpha=0.35)
    if prompts is not None:
        _draw_prompts(axes[2], prompts)
    axes[2].set_title("UltraSAM")
    axes[2].axis("off")

    # Panel 4: Image + prompts only
    axes[3].imshow(image)
    if prompts is not None:
        _draw_prompts(axes[3], prompts)
    axes[3].set_title("Prompts")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _overlay_mask(
    ax: plt.Axes, mask: np.ndarray, color: str = "red", alpha: float = 0.3
) -> None:
    """Overlay a binary mask on a matplotlib axis."""
    color_map = {
        "red": [1, 0, 0],
        "green": [0, 1, 0],
        "blue": [0, 0, 1],
        "yellow": [1, 1, 0],
    }
    rgb = color_map.get(color, [1, 0, 0])
    overlay = np.zeros((*mask.shape[:2], 4), dtype=np.float32)
    binary = mask.astype(bool)
    overlay[binary] = [*rgb, alpha]
    ax.imshow(overlay)
