import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDiceLoss(nn.Module):
    """Dice loss for binary class."""
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'Unexpected reduction {self.reduction}')

class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation."""
    def __init__(self, weight=None, ignore_index=None, smooth=1, p=2, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        # Apply softmax to get probabilities for each class
        predict = F.softmax(predict, dim=1)

        # One-hot encode the target masks
        target = F.one_hot(target, num_classes=predict.shape[1]).permute(0, 3, 1, 2).float()

        dice = BinaryDiceLoss(smooth=self.smooth, p=self.p, reduction=self.reduction)
        total_loss = 0
        
        # Calculate Dice loss for each class
        for i in range(predict.shape[1]):  # Loop through each class
            if i != self.ignore_index:  # Skip ignored index if provided
                dice_loss = dice(predict[:, i], target[:, i])

                # Apply class weights if provided
                if self.weight is not None:
                    assert self.weight.shape[0] == predict.shape[1], \
                        f'Expected weight shape [{predict.shape[1]}], got [{self.weight.shape[0]}]'
                    dice_loss *= self.weight[i]

                total_loss += dice_loss

        return total_loss / predict.shape[1]
