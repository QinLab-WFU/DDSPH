from argparse import Namespace

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, arccos, cos

from utils import gen_triplets


class DDSPHLoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.C = args.nclass
        self.N = args.N  # proxies_per_class
        self.r = args.r  # ration which determines how many proxies are selected
        self.scale = args.scale

        self.proxies = nn.Parameter(torch.Tensor(args.output_dim, self.C * self.N))  # K x C*N
        nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))

        proxy_single_labels = torch.tensor(np.repeat(np.arange(self.C), self.N))  # C*N
        proxy_labels = F.one_hot(proxy_single_labels, self.C).float()  # C*N x C
        self.register_buffer("proxy_labels", proxy_labels)

        self.triplet = TripletLoss()

    def scale_mask_softmax(self, tensor, mask, softmax_dim, scale=1.0):
        scale_mask_exp_tensor = torch.exp(tensor * scale) * mask.detach()
        scale_mask_softmax_tensor = scale_mask_exp_tensor / (
            1e-8 + torch.sum(scale_mask_exp_tensor, dim=softmax_dim)
        ).unsqueeze(softmax_dim)
        return scale_mask_softmax_tensor

    def fwd_cls_loss(self, batch, labels, centers):
        # 1) constructing directed similarity graph
        similarity = batch.matmul(centers)  # S: B x K, K x C*N -> B x C*N

        # 2) relation-guided sub-graph construction

        # positive_mask = torch.eq(
        #     single_labels.view(-1, 1) - self.proxy_single_labels.view(1, -1), 0.0
        # ).float()
        # positive_mask = (labels @ self.proxy_labels.T > 0).float()  # S^pos: B x C*N
        #-=-=-=-=-=-=-=-=-=-=--=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-==-=-=-=-=-=-=-=-=-=-=--
        if self.args.posmask == "soft":
            pos = labels @ self.proxy_labels.T  # (B, K), = #shared labels
            num_labels = labels.sum(dim=1, keepdim=True).clamp(min=1.0)
            positive_mask = (pos / num_labels).clamp(max=1.0)  # soft mask in [0,1]
        elif self.args.posmask == "hard":
            positive_mask = (labels @ self.proxy_labels.T > 0).float()
        else:
            positive_mask = torch.zeros(labels.size(0), self.C * self.N, device=labels.device)

        topk = math.ceil(self.r * self.C * self.N)
        _, indices = torch.topk(similarity + 1000 * positive_mask, topk, dim=1)  # "1000*" aims to rank faster

        if self.args.use_rlp:
            mask = torch.zeros_like(similarity).scatter_(1, indices, 1)
            prob_a = mask * similarity  # W: B x C*N

            # 3) revere label propagation (including classification process)
            logits = torch.matmul(prob_a, self.proxy_labels) / (mask.sum(dim=1, keepdim=True) + 1e-6)  # Z: B x C
        else:
            logits = similarity @ self.proxy_labels / (similarity.size(1))
        # logits_mask = 1 - torch.eq(logits, 0.0).float()  # M: B x C
        # predict = self.scale_mask_softmax(logits, logits_mask, 1, self.scale)
        # classification loss: Eq. 7
        # return torch.mean(torch.sum(-labels * torch.log(predict + 1e-20), dim=1))  # 鼓励单标签
        # 另一版多标签
        # predict = torch.sigmoid(logits)
        # return F.binary_cross_entropy(predict, labels)
        # 多标签，但是不够集中
        return F.binary_cross_entropy_with_logits(logits, labels)

    def forward(self, img_embs, txt_embs, labels):
        centers = F.normalize(self.proxies, p=2, dim=0)
        img_embs = F.normalize(img_embs)
        txt_embs = F.normalize(txt_embs)
        L_s = self.fwd_cls_loss(img_embs, labels, centers) + self.fwd_cls_loss(txt_embs, labels, centers)

        gate = torch.sigmoid(img_embs @ txt_embs.T).mean(dim=1, keepdim=True)
        it_embs = gate * img_embs + (1 - gate) * txt_embs
        L_s += self.fwd_cls_loss(it_embs, labels, centers)

        # 4) regularization on proxies: Eq. 11
        center_sims = centers.t().matmul(centers)  # C*N x C*N
        center_logits = torch.matmul(center_sims, self.proxy_labels)  # C*N x C
        if self.args.proxy_reg_type == "ce":
            L_p = F.cross_entropy(center_logits, torch.argmax(self.proxy_labels, dim=1)) # 单标签
        elif self.args.proxy_reg_type == "bce":
            L_p = F.binary_cross_entropy_with_logits(center_logits, self.proxy_labels) # 改为多标签
        elif self.args.proxy_reg_type == "sigmoid":
            L_p = -(self.proxy_labels * torch.logsigmoid(center_logits)).mean()
        elif self.args.proxy_reg_type == "l2":
            L_p = (center_sims ** 2).mean()
        else:
            L_p = torch.tensor(0.0, device=center_sims.device)

        L_t = self.triplet(img_embs, labels) + self.triplet(txt_embs, labels)
        L_t += self.triplet(img_embs, labels, txt_embs) + self.triplet(txt_embs, labels, img_embs)

        return L_s, L_p, L_t

def proxy_align_loss(img, txt, centers):
    sim_i = img @ centers
    sim_t = txt @ centers
    return F.mse_loss(sim_i, sim_t)

class TripletLoss(nn.Module):
    def __init__(self, margin=0.25, verbose=False, normalized=False):
        super().__init__()
        self.margin = margin
        self.verbose = verbose
        self.normalized = normalized

    def forward(self, embeddings, labels, embed=None):
        if embed is None:
            embed = embeddings
        sim_mat = embeddings @ embed.T

        anc_idxes, pos_idxes, neg_idxes = gen_triplets(labels)

        S_ap = sim_mat[anc_idxes, pos_idxes]
        S_an = sim_mat[anc_idxes, neg_idxes]

        losses = F.relu(S_an - S_ap + self.margin)

        # mask = (losses > 0) & (S_ap - S_an >= 0)  # semi-hard
        mask = losses > 0  # all

        N = mask.sum()
        HN_ratio = 0

        if N == 0:
            loss = torch.tensor(0.0, requires_grad=True, device=sim_mat.device)
        else:
            loss = losses[mask].mean()

            N_hard = (mask & (S_an >= S_ap)).float().sum()
            HN_ratio = N_hard / N

        if not self.verbose:
            return loss

        # return loss, N, HN_ratio
        return loss

def multilabel_info_nce(img, txt, labels, temperature=0.07):
    """
    img: (B, D)
    txt: (B, D)
    labels: (B, C) multi-hot
    """
    img = F.normalize(img, dim=1)
    txt = F.normalize(txt, dim=1)

    sim = img @ txt.t() / temperature  # (B, B)

    # 正样本：共享 ≥1 标签
    pos_mask = (labels @ labels.t() > 0).float()

    # 防止全 0
    pos_mask.fill_diagonal_(0)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    loss = -(pos_mask * log_prob).sum() / (pos_mask.sum() + 1e-6)
    return loss

if __name__ == "__main__":
    from _utils import gen_test_data

    B, C, K = 120, 10, 8
    e, t, l = gen_test_data(B, C, K)

    criterion = ProxyGMLLoss(Namespace(n_bits=K, n_classes=C))
    loss = criterion(e, l)
    print(loss)
