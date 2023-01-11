import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def binary_focal_loss(pred, target, gamma=2.0, alpha=0.5, reduction='mean'):

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
                reduction_override=None,
                ignore_index=255):
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
               "The shape of pred doesn't match the shape of target"

        original_shape = pred.shape

        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()

        if original_shape == target.shape:
            # target with shape [B, C, d_1, d_2, ...]
            # transform it's shape into [N, C]
            # [B, C, d_1, d_2, ...] -> [C, B, d_1, d_2, ..., d_k]
            target = target.transpose(0, 1)
            # [C, B, d_1, d_2, ..., d_k] -> [C, N]
            target = target.reshape(target.size(0), -1)
            # [C, N] -> [N, C]
            target = target.transpose(0, 1).contiguous()
        else:
            # target with shape [B, d_1, d_2, ...]
            # transform it's shape into [N, ]
            target = target.view(-1).contiguous()
            valid_mask = (target != ignore_index).view(-1, 1)
            # avoid raising error when using F.one_hot()
            target = torch.where(target == ignore_index, target.new_tensor(0),
                                 target)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * binary_focal_loss(
            pred, target, weight, reduction=reduction)
        return loss

