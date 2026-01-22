import torch

def dice_coef(preds, targets, eps=1e-6):
    """
    preds: (B,1,H,W) logits or probs
    targets: (B,H,W) 0/1
    """
    preds = (preds > 0.5).float()
    targets = targets.float().unsqueeze(1)

    inter = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()

def iou_coef(preds, targets, eps=1e-6):
    preds = (preds > 0.5).float()
    targets = targets.float().unsqueeze(1)

    inter = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()
