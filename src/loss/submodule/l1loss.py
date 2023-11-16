"""
    CompletionFormer
    ======================================================================

    L1 loss implementation
"""


import torch
import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, pred, gt):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)
        pred = torch.clamp(pred, min=0, max=self.args.max_depth)

        mask = (gt > self.t_valid).type_as(pred).detach()

        d = torch.abs(pred - gt) * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss

class L1Loss_w_unc(nn.Module):
    def __init__(self, args):
        super(L1Loss_w_unc, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, pred, gt, uncertainty):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)
        pred = torch.clamp(pred, min=0, max=self.args.max_depth)

        y_logvar = uncertainty
        reg_unc = torch.log(uncertainty+1e-16)

        mask = (gt > self.t_valid).type_as(pred).detach()
        d = torch.abs(pred - gt) 
        d = 20 * y_logvar * d - reg_unc
        d = d * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss
