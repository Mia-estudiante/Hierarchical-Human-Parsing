from itertools import filterfalse as ifilterfalse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn import BCELoss
import utils.aaf.losses as lossx
#weights
# ATR training
# [0.85978634, 1.19630769, 1.02639146, 1.30664970, 0.97220603, 1.04885815,
#  1.01745278, 1.01481690, 1.27155077, 1.12947663, 1.13016390, 1.06514227,
#  1.08384483, 1.08506841, 1.09560942, 1.09565198, 1.07504567, 1.20411509]

#CCF
# [0.82073458, 1.23651165, 1.0366326,  0.97076566, 1.2802332,  0.98860602,
#  1.29035071, 1.03882453, 0.96725283, 1.05142434, 1.0075884,  0.98630539,
#  1.06208869, 1.0160915,  1.1613597,  1.17624919, 1.1701143,  1.24720215]

#PPSS
# [0.89680465, 1.14352656, 1.20982646, 0.99269248,
#  1.17911144, 1.00641032, 1.47017195, 1.16447113]

#Pascal
# [0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914]

#Lip
# [0.7602572,  0.94236198, 0.85644457, 1.04346266, 1.10627293, 0.80980162,
#  0.95168713, 0.8403769,  1.05798412, 0.85746254, 1.01274366, 1.05854692,
#  1.03430773, 0.84867818, 0.88027721, 0.87580925, 0.98747462, 0.9876475,
#  1.00016535, 1.00108882]

class ABRLovaszLoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(ABRLovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn
    

class ABRLovaszCELoss(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, ignore_index=None, only_present=True):
        super(ABRLovaszCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914])

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss_ce = self.criterion(pred, targets[0])

        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        # loss_hb_ce = self.criterion2(pred_hb, targets[1].long())

        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        # loss_fb_ce = self.criterion2(pred_fb, targets[2].long())

        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        total_loss =  loss_ce + loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn
        
        return total_loss



class LovaszSoftmaxLoss(nn.Module):
    """Lovasz loss for Deep Supervision"""

    def __init__(self, ignore_index=None, only_present=False, per_image=False):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.per_image = per_image
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight)

    def forward(self, preds, targets):
        h, w = targets.size(1), targets.size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        if self.per_image:
            loss = mean(lovasz_softmax_flat(*flatten_probas(pre.unsqueeze(0), tar.unsqueeze(0), self.ignore_index),
                                            only_present=self.only_present) for pre, tar in zip(pred, targets))
        else:
            loss = lovasz_softmax_flat(*flatten_probas(pred, targets, self.ignore_index),
                                       only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets)
        return loss + 0.4 * loss_dsn


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat_ori(*flatten_probas_ori(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat_ori(*flatten_probas_ori(probas, labels, ignore), classes=classes)
    return loss

def lovasz_softmax_flat_ori(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas_ori(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def lovasz_softmax_flat(preds, targets, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      :param preds: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      :param targets: [P] Tensor, ground truth labels (between 0 and C - 1)
      :param only_present: average only on classes present in ground truth
    """
    if preds.numel() == 0:
        # only void pixels, the gradients should be 0
        return preds * 0.

    C = preds.size(1)
    losses = []
    for c in range(C):
        fg = (targets == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - preds[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probas(preds, targets, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = preds.size()
    preds = preds.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    targets = targets.view(-1)
    if ignore is None:
        return preds, targets
    valid = (targets != ignore)
    vprobas = preds[valid.nonzero().squeeze()]
    vlabels = targets[valid]
    return vprobas, vlabels

# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def mean(l, ignore_nan=True, empty=0):
    """
    nan mean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def isnan(x):
    return x != x
