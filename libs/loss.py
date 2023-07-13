import torch
import torch.nn as nn
import torch.nn.functional as F


class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None):

        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()
        diff = torch.abs(target - pred)
        diff = diff[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0*delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.sum()/diff.numel()
        return loss


class RMSELog(nn.Module):
    def __init__(self):
        super(RMSELog, self).__init__()

    def forward(self, target, pred, mask=None):
        #assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()
        target = target[valid_mask]
        pred = pred[valid_mask]
        log_error = torch.abs(torch.log(target / (pred + 1e-12)))
        loss = torch.sqrt(torch.mean(log_error**2))
        return loss
