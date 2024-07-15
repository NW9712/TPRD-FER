import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy
import numpy as np
import math
import torch.nn.functional as F
from scipy import special

eps = 1e-12

class LabelSmoothing_CrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def linear_combination(self, x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.linear_combination(loss / n, nll, self.epsilon)


def kl_loss_compute(pred, soft_targets, reduce=True):
    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def js_loss_compute(pred, soft_targets, reduce=True):
    kl1 = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)
    kl2 = F.kl_div(F.log_softmax(soft_targets, dim=1),F.softmax(pred, dim=1),reduce=False)
    js = 0.5 * (kl1 + kl2)
    if reduce:
        return torch.mean(torch.sum(js, dim=1))
    else:
        return torch.sum(js, 1)

def Distance_squared(x, y, featdim=1):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    d = dist.clamp(min=eps)
    d[torch.eye(d.shape[0]) == 1] = eps
    return d

def Distance(x, y, featdim=1):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    d = dist.clamp(min=eps)
    d = d**0.5
    # d = (d-d.min())/(d.max()-d.min())
    d[torch.eye(d.shape[0]) == 1] = eps
    return d

def Distance_norm(x, y, featdim=1):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    d = dist.clamp(min=eps)
    d = d**0.5
    d = (d-d.min())/(d.max()-d.min())
    d[torch.eye(d.shape[0]) == 1] = eps
    return d

def CalPairwise(dist,alpha = 1.0):
    dist[dist < 0] = 0
    Pij = torch.exp(-(dist/alpha))
    return Pij


def refining_loss_compute(feat1, feat2, alpha = 1.0):
    q1 = CalPairwise(Distance_squared(feat1, feat1), alpha)
    q2 = CalPairwise(Distance_squared(feat2, feat2), alpha)
    # q1 = CalPairwise(Distance(feat1, feat1), alpha)
    # q2 = CalPairwise(Distance(feat2, feat2), alpha)
    return - (q1 * torch.log(q2 + eps)).mean()

    # # print('q1')
    # # print(q1)
    # # print(-torch.log(q2))
    # return - (q1 * alpha * torch.log(q2 + eps)).sum() / (feat1.size(0) * (7-1))  # num_class = 7

    # q1 = Distance_squared(feat1, feat1)
    # q2 = Distance_squared(feat2, feat2)
    # return torch.log(1+(1/q1*(1+q2))**2).mean()
    # return (q1 * torch.log(q1 + eps) - q1 * torch.log(q2 + eps)).mean()

# def refining_loss_compute(feat1, feat2, alpha = 1.0):
#     q1 = CalPairwise(Distance_norm(feat1, feat1)/Distance_norm(feat2,feat2), alpha)
#     q2 = CalPairwise(Distance_squared(feat2, feat2), alpha)
#
#     return - (q1 * torch.log(q2 + eps)).mean()

# def refining_loss_compute(feat1, feat2, alpha = 1.0):
#     q1 = Distance(feat1, feat1)
#     q2 = Distance(feat2, feat2)
#     # q3 = Distance(feat2, feat2)
#     kl = F.kl_div(F.log_softmax(q1, dim=1),F.softmax(q2, dim=1),reduce=False)
#     # kl = kl*(q3)
#     return torch.mean(torch.sum(kl, dim=1))

    # return (q3 * torch.log(q2 + eps) - q3 * torch.log(q1 + eps)).mean()

# def refining_loss_compute(feat1, feat2, alpha = 1.0):
#     q1 = Distance_norm(feat1, feat1)
#     q2 = Distance_norm(feat2, feat2)
#     q3 = Distance(feat2, feat2)
#     kl = F.kl_div(F.log_softmax(q2, dim=1),F.softmax(q1, dim=1),reduce=False)
#
#     return (F.log_softmax(q3,dim=1) * torch.log(F.softmax(q2,dim=1) + eps) - F.log_softmax(q3,dim=1) * torch.log(F.softmax(q1,dim=1) + eps)).mean()



class Soft_CrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes = 7, reduction = 'mean'):
        super(Soft_CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.num_classes = num_classes
        self.eps = 1e-8

    def forward(self, logits, targets):
        # logits: [N, C], targets: [N,C]
        if logits.dim() != 2 and targets.dim() != 2:
            raise Exception
        targets = F.normalize(targets, p=1, dim=-1)
        logits_norm = torch.nn.functional.softmax(logits, dim=1)
        # logits_argsort = torch.argsort(logits_norm, dim = 1, descending = True)
        loss = targets * torch.log(logits_norm + self.eps)
        if self.reduction == 'mean':
            loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise NotImplementedError
        return loss

class WeakSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.5, contrast_mode='all'):
        super(WeakSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        # self.base_temperature = base_temperature
        self.similarity_function = self._get_similarity_function(use_cosine_similarity=True)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        v = torch.div(
            v,
            # torch.matmul(anchor_feature, contrast_feature.T),
            v.size(0))
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        labels = labels.contiguous().view(1, -1)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        logits = torch.div(
            self.similarity_function(anchor_feature, contrast_feature),
            # torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        positives = torch.scatter(
            torch.zeros_like(logits),
            1,
            torch.cat([torch.arange(batch_size * 1, batch_size * anchor_count), torch.arange(batch_size * 1)], dim=0).view(-1, 1).to(device),
            1
        )

        targets_matrix = torch.cat([labels, labels], dim=1)
        targets_matrix = targets_matrix.repeat(2 * batch_size,1)
        targets_matrix_t = targets_matrix.permute(1, 0)

        negative = (~ torch.eq(targets_matrix, targets_matrix_t))

        # compute log_prob
        exp_logits = torch.exp(logits)

        # log_prob = logits - torch.log((exp_logits * negative).sum(1, keepdim=True)) / torch.count_nonzero(negative, dim =1).unsqueeze(dim=1)
        log_prob = logits - torch.log((exp_logits * negative).sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (positives * log_prob).sum(1) / positives.sum(1)

        # print(mean_log_prob_pos.shape)
        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print(loss.shape)

        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.similarity_function = self._get_similarity_function(use_cosine_similarity=True)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        v = torch.div(
            v,
            # torch.matmul(anchor_feature, contrast_feature.T),
            v.size(0))
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)
        anchor_dot_contrast = torch.div(
            self.similarity_function(anchor_feature, contrast_feature),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss





