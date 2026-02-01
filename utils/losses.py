"""Loss functions for TransUNet and UltraSAM training."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, n_classes=2, smooth=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = 1 - (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f'predict {inputs.size()} & target {target.size()} shape do not match'
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


class BinaryDiceLoss(nn.Module):
    """Binary dice loss for sigmoid outputs."""
    def __init__(self, smooth=1.0, eps=1.0):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, target):
        """
        pred: (B, H, W) logits or probabilities
        target: (B, H, W) binary
        """
        pred = torch.sigmoid(pred)
        pred = pred.flatten(1)
        target = target.flatten(1).float()
        intersection = (pred * target).sum(1)
        union = pred.sum(1) + target.sum(1)
        dice = (2.0 * intersection + self.smooth) / (union + self.eps)
        return (1.0 - dice).mean()


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        """
        pred: (B, H, W) logits
        target: (B, H, W) binary
        """
        pred = pred.flatten(1)
        target = target.flatten(1).float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-bce)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


class CombinedTransUNetLoss(nn.Module):
    """Combined CE + Dice loss for TransUNet (0.5 each)."""
    def __init__(self, n_classes=2):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes=n_classes)

    def forward(self, pred, target):
        loss_ce = self.ce_loss(pred, target.long())
        loss_dice = self.dice_loss(pred, target, softmax=True)
        return 0.5 * loss_ce + 0.5 * loss_dice


class CombinedUltraSAMLoss(nn.Module):
    """Combined Focal + Dice + IoU MSE loss for UltraSAM."""
    def __init__(self, focal_weight=20.0, dice_weight=1.0, iou_weight=1.0):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
        self.dice_loss = BinaryDiceLoss()
        self.iou_loss = nn.MSELoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

    def forward(self, pred_masks, gt_masks, pred_iou, gt_iou):
        loss_focal = self.focal_loss(pred_masks, gt_masks)
        loss_dice = self.dice_loss(pred_masks, gt_masks)
        loss_iou = self.iou_loss(pred_iou, gt_iou)
        return (self.focal_weight * loss_focal +
                self.dice_weight * loss_dice +
                self.iou_weight * loss_iou)
