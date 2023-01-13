import torch
import torch.nn.functional as F
import cv2
import torch.nn as nn
from timm.utils import accuracy, AverageMeter
import numpy as np
import itertools
import math

# ----------------------------------------------------------------------------------------------------------------------
class TVLoss(torch.nn.Module):
    def __init__(self, attn_dim=14):
        super(TVLoss, self).__init__()

        self.attn_dim = attn_dim

    def forward(self, inp):
        x = inp['dot_qk']
        bs, nh, _, _ = x.shape
        x = x[:, :, 0, 1:].reshape(bs, nh, self.attn_dim, self.attn_dim)

        h_tv = 0.5 * torch.pow((x[:, :, :-1, :] - x[:, :, 1:, :]), 2).sum(dim=[2, 3])
        w_tv = 0.5 * torch.pow((x[:, :, :, :-1] - x[:, :, :, 1:]), 2).sum(dim=[2, 3])
        return torch.mean(h_tv + w_tv)

# ----------------------------------------------------------------------------------------------------------------------
class DisReg(torch.nn.Module) :
    def __init__(self) :
        super(DisReg, self).__init__()
        idxs = np.array([i for i in itertools.permutations(range(6), 2)])
        self.id0 = idxs[:, 0]
        self.id1 = idxs[:, 1]

    def forward(self, inp) :
        bs, L, nh, dim = inp['head_out'].shape
        x = inp['head_out'].permute(0, 2, 1, 3).reshape(bs, nh, L * dim)
        loss = F.cosine_similarity(x[:, self.id0], x[:, self.id1], dim=-1)

        return torch.mean(loss)


# ----------------------------------------------------------------------------------------------------------------------
class CComponents(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def opencv_cc(self, x):
        if len(x.shape) == 2:
            x = x[:, :, None]
        x = ((x > 0) * 1).astype('uint8')
        n_blobs, lb = cv2.connectedComponents(x, connectivity=8, ltype=cv2.CV_16U)
        return n_blobs, lb

    @torch.no_grad()
    def forward(self, x):
        device = x.device
        x = x.detach().cpu().numpy()
        bs, nh, _ = x.shape

        batch_idx = []
        head_idx = []
        spatial_idx = []

        for b in range(bs):
            for h in range(nh):
                n_blobs, cc = self.opencv_cc(x[b, h].reshape(14, 14))
                for k in range(1, n_blobs):
                    batch_idx.append(b)
                    head_idx.append(h)
                    spatial_idx.append(torch.tensor((cc == k) * 1.0).reshape(-1).long())

        batch_idxs = torch.tensor(batch_idx).to(device)
        head_idx = torch.tensor(head_idx).to(device)
        spatial_idx = torch.stack(spatial_idx).to(device)

        return batch_idxs, head_idx, spatial_idx

# ========================= Blob loss ================================

class BlobLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.cc = CComponents()

    def forward(self, inp):
        with torch.cuda.amp.autocast(enabled=False):
            x = inp['dot_qk'][:, :, 0, 1:]
            bs = x.shape[0]
            nh = x.shape[1]

            # Detect blobs
            m = torch.mean(x, dim=-1, keepdim=True)

            # Compute connected components
            with torch.no_grad():
                B_mask = (x > m).long()
                b_id, h_id, mask = self.cc(B_mask)

            # Activation
            x = torch.relu(x - m) + torch.tensor(1e-9)

            # Compute probabilities
            p_u = torch.sum(x[b_id, h_id] * mask, dim=-1)
            B = torch.sum(x[b_id, h_id] * B_mask[b_id, h_id], dim=-1)
            p_n = p_u / B

            # Compute entropy
            H = -1 * (p_n * torch.log(p_n))
            loss = torch.sum(H) / (bs * nh)

        if not math.isfinite(loss):
            return None
        else:
            return loss