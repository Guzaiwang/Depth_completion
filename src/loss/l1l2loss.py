"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NLSPNLoss implementation
"""


from . import BaseLoss, BaseLoss_w_unc, BaseLoss_w_confidence
import torch
import torch.nn.functional as F


class L1L2Loss(BaseLoss):
    def __init__(self, args):
        super(L1L2Loss, self).__init__(args)

        self.loss_name = []
        self.t_valid = 0.0001

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output):
        loss_val = []

        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue

            pred = output['pred']
            gt = sample['gt']

            loss_tmp = 0.0
            if loss_type in ['L1', 'L2']:
                loss_tmp += loss_func(pred, gt) * 1.0
            else:
                raise NotImplementedError

            loss_tmp = loss['weight'] * loss_tmp
            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val

class L1L2wUnC(BaseLoss_w_unc):
    def __init__(self, args):
        super(L1L2wUnC, self).__init__(args)

        self.loss_name = []
        self.t_valid = 0.0001

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output):
        loss_val = []

        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue

            pred = output['pred']
            B, _, H, W = pred.shape
            gt = sample['gt']
            gt = F.interpolate(gt, size=[H, W])
            unc = output['uncertainty']

            loss_tmp = 0.0
            if loss_type in ['L1', 'L2']:
                loss_tmp += loss_func(pred, gt, unc) * 1.0
            else:
                raise NotImplementedError

            loss_tmp = loss['weight'] * loss_tmp
            loss_val.append(loss_tmp)


        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val


class L1L2wConfidence(BaseLoss_w_confidence):
    def __init__(self, args):
        super(L1L2wConfidence, self).__init__(args)

        self.loss_name = []
        self.t_valid = 0.0001

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output, first_output):
        loss_val = []

        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue

            pred = output['pred']
            gt = sample['gt']
            confidence = output['uncertainty']

            loss_tmp = 0.0
            if loss_type in ['L1', 'L2']:
                loss_tmp += loss_func(pred, gt, confidence) * 1.0
            else:
                raise NotImplementedError
            
            first_pred = first_output['pred'].detach()
            first_confidence = first_output['uncertainty'].detach()


            train_consistancy = torch.abs(pred - first_pred) / (torch.abs(pred)+ 0.0000001)
            loss_consistancy = torch.abs((confidence - first_confidence) / first_confidence)

            # compute consistancy loss
            consistancy_loss = torch.nn.functional.relu( - torch.sign(confidence - first_confidence)* train_consistancy + loss_consistancy)
            consistancy_loss = consistancy_loss.mean()
            loss_tmp += consistancy_loss

            loss_tmp = loss['weight'] * loss_tmp
            loss_val.append(loss_tmp)


        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val

