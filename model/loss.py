import torch
import torch.nn.functional as F
import torch.nn as nn


def bce_iou_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    return (weighted_bce + weighted_iou).mean()


def dice_bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (2. * inter + 1) / (union + 1)

    return (bce + iou).mean()


def tversky_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    pred = torch.sigmoid(pred)

    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

    return (1 - Tversky) ** gamma


def tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred = torch.sigmoid(pred)

    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

    return bce + (1 - Tversky) ** gamma


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1)  # (N, C, *)
        target = target.view(N, 1, -1)  # (N, 1, *)

        predict = F.softmax(predict, dim=1)  # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


def structure_loss_with_ual(pred, mask):
    return structure_loss(pred, mask) + 0.5 * cal_ual(pred, mask)


class Bce_iou_loss(nn.Module):

    def __init__(self):
        super(Bce_iou_loss, self).__init__()

    def forward(self, pred, mask):
        weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

        pred = torch.sigmoid(pred)
        inter = pred * mask
        union = pred + mask
        iou = 1 - (inter + 1) / (union - inter + 1)

        weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
        weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

        return (weighted_bce + weighted_iou).mean()

def structure_loss_multiclass(pred, mask):
    """
    Structure loss for multiclass segmentation
    pred: [B, C, H, W] - model predictions (logits)
    mask: [B, 1, H, W] - ground truth with class indices (0, 1, 2, 3)
    """
    smooth = 1e-8
    B, C, H, W = pred.shape
    
    # 1) Convert mask to one-hot: shape → [B, C, H, W]
    labels      = mask.squeeze(1).long()                     # [B, H, W]
    mask_onehot = F.one_hot(labels, num_classes=C)           # [B, H, W, C]
    mask_onehot = mask_onehot.permute(0, 3, 1, 2).float()    # [B, C, H, W]

    # 2) Compute per-class “edge” weights via large average-pool:
    #    Boundaries in any class will have higher weight.
    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(mask_onehot, kernel_size=31, stride=1, padding=15)
        - mask_onehot
    )

    # 3) Gather the weight corresponding to the true class at each pixel:
    #    weight_pixel[b, h, w] = weight[b, labels[b,h,w], h, w]
    weight_pixel = weight.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B, H, W]

    # 4) Weighted cross-entropy (per-pixel reduction):
    #    - reduction='none' → keep per-pixel CE
    #    - multiply by weight_pixel
    #    - sum over H×W, then normalize by total weight
    ce_map       = F.cross_entropy(pred, labels, reduction='none')  # [B, H, W]
    weighted_ce  = (weight_pixel * ce_map).sum((1, 2)) / (weight_pixel.sum((1, 2)) + smooth)           # [B]

    # 5) Weighted Dice loss (vectorized over classes):
    pred_soft    = F.softmax(pred, dim=1)                         # [B, C, H, W]
    #   intersection = ∑ weight * P * GT
    #   union        = ∑ weight * (P + GT)
    intersection = (weight * pred_soft * mask_onehot).sum((2, 3))  # [B, C]
    union        = (weight * (pred_soft + mask_onehot)).sum((2, 3))  # [B, C]
    dice_score   = (2 * intersection + smooth) / (union + smooth)   # [B, C]
    dice_loss    = 1 - dice_score                                   # [B, C]
    dice_loss    = dice_loss.mean(dim=1)                            # average over C → [B]

    # 6) Combine CE + Dice and average over the batch
    loss = (weighted_ce + dice_loss).mean()

    return loss