import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def binary_focal_loss(pred, target, gamma=2.0, alpha=0.5, reduction='mean'):
    assert pred.size() == target.size() and target.numel() > 0

    logpt = F.sigmoid(pred)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())

    if alpha is not None:
        at = alpha.gather(0, target.data.view(-1))
        logpt = logpt * Variable(at)

    loss = -1 * (1-pt)**gamma * logpt

    if reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()


@LOSSES.register_module
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean', loss_weight=1.0, use_sigmoid=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.use_sigmoid=use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_name = 'loss_focal'
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = torch.Tensor([alpha, 1-alpha])

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * binary_focal_loss(
            pred, target, weight, reduction=reduction)
        return loss

