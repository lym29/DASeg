import numpy as np
import torch
import pyflann
import torch.nn as nn
import torch.nn.functional as F


def build_extended_map(p_size, x):
        n, c, h, w = x.size()
        step = p_size // 2
        y = torch.zeros((n, c, h + 2 * step, w + 2 * step))
        y[:, :, step:step + h, step:step + w] = x
        y[:, :, step:step + h, :step] = x[:, :, :, 0:1].expand((n, c, h, step))
        y[:, :, step:step + h, step + w:step + 2 * w] = x[:, :, :, -1:w + 2 * step].expand((n, c, h, step))
        y[:, :, :step, :] = y[:, :, step:step + 1, :].expand((n, c, step, w + 2 * step))
        y[:, :, h + step:, :] = y[:, :, h + step - 1:h + step, :].expand((n, c, step, w + 2 * step))
        return y


def bds_vote(ref, nnf_st, nnf_ts, p_size=3, ignore_label=255):
    """
    Reconstructs an image or feature map by bidirectional
    similarity voting
    """
    batch_size, c, _, _ = ref.size()
    _, _, sh, sw = nnf_st.size()
    _, _, th, tw = nnf_ts.size()
    step = p_size // 2

    ref = F.upsample(ref.float(), size=(sh, sw), mode='nearest')

    num_s = sh * sw
    num_t = th * tw

    ex_ref = build_extended_map(p_size, ref)
    ex_guide = torch.zeros(batch_size, c, th + 2 * step, tw + 2 * step)
    ex_weight = torch.zeros(batch_size, c, th + 2 * step, tw + 2 * step)

    for n in range(batch_size):
        # S to T Complete
        for i in range(sh):
            for j in range(sw):
                t_i, t_j = nnf_st[n, :, i, j]
                ex_guide[n, :, t_i:t_i + p_size, t_j:t_j + p_size] += ex_ref[n, :, i:i + p_size, j:j + p_size] / num_s
                ex_weight[n, :, t_i:t_i + p_size, t_j:t_j + p_size] += 1 / num_s
        # T to S Cohere
        for i in range(th):
            for j in range(tw):
                s_i, s_j = nnf_ts[n, :, i, j]
                ex_guide[n, :, i:i + p_size, j:j + p_size] += ex_ref[n, :, s_i:s_i + p_size, s_j:s_j + p_size] / num_t
                ex_weight[n, :, i:i + p_size, j:j + p_size] += 1 / num_t

    ex_weight[ex_weight < 1e-8] = 1
    ex_guide /= ex_weight
    ex_guide[ex_weight < 1e-8] = ignore_label

    return ex_guide[:, :, step:-step, step:-step]


def build_prob_map(labels, num_classes):
    n, _, h, w = labels.size()
    l = labels.squeeze(1)
    p = torch.zeros(n, num_classes, h, w)
    for c in range(num_classes):
        p[:, c, :, :][l == c] = 1
    return p

def pairwise_dist(pts, q_pts):
    r_p = torch.sum(pts * pts, dim=1, keepdim=True)  # (N,1)
    r_q = torch.sum(q_pts * q_pts, dim=1, keepdim=True)  # (M,1)
    mul = torch.matmul(q_pts, pts.transpose(1,0))         # (M,N)
    dist = r_p - 2 * mul + r_q.permute(1,0)       # (M,N)
    return dist

def nearest_neighbour(pts, q_pts):
    dist = pairwise_dist(pts, q_pts)
    nn = torch.argmin(dist, dim=1) #(M, 1)
    nnd = torch.min(dist, dim=1)[0]
    return nn, nnd

class PatchMatch(nn.Module):
    def __init__(self, s, t, patch_size=3):
        super(PatchMatch, self).__init__()
        self.source_map = s
        self.target_map = t
        self.f_dim = self.source_map.shape[2]
        self.patch_size = patch_size

        n, c, h, w = s.size()

        self.batch_size = n
        self.nnf = torch.zeros((n, 2, h, w), dtype=int)  # The NNF
        self.nnd = torch.zeros(n, 1, h, w)  # The NNF distance map

    def build_pts_set_for_flann(self, x):
        patch_size = self.patch_size
        n, c, h, w = x.size()
        y = build_extended_map(patch_size, x)
        # build feat tensor for flann
        ex_feat = torch.zeros(n, c, patch_size * patch_size, h, w)  # extended feature map
        for i in range(patch_size):
            for j in range(patch_size):
                pn = i * patch_size + j
                ex_feat[:, :, pn, :, :] = y[:, :, i:i + h, j:j + w]

        temp = ex_feat.reshape((n, c * patch_size * patch_size, h, w))
        point_set = temp.reshape((n, c * patch_size * patch_size, h * w))

        return point_set

    def find_nnf(self):
        _, _, th, tw = self.target_map.size()
        _, _, sh, sw = self.source_map.size()
        flann = pyflann.FLANN()
        pts_tensor = self.build_pts_set_for_flann(self.target_map)
        q_pts_tensor = self.build_pts_set_for_flann(self.source_map)
        for n in range(self.batch_size):
            pts = pts_tensor[n, :, :].detach().transpose(1,0)
            q_pts = q_pts_tensor[n, :, :].detach().transpose(1,0)
            result_id, dists = nearest_neighbour(pts, q_pts)
            idy, idx = result_id // tw, result_id % tw
            self.nnf[n, 0, :, :] = idy.reshape(sh, sw)
            self.nnf[n, 1, :, :] = idx.reshape(sh, sw)
            self.nnd[n, 0, :, :] = dists.reshape(sh, sw)

    def forward(self):
        self.find_nnf()
        return self.nnf, self.nnd
    
def find_NBB(nnf_st, nnf_ts, nnd_st, label, prob, ignore_label = 255):
    batch_size, _, sh, sw = nnf_st.size()
    _, _, sh, sw = nnf_st.size()
    _, _, th, tw = nnf_ts.size()

    label = F.upsample(label.float(), size=(sh, sw), mode='nearest')
    label = label.long()
    prob = F.upsample(prob, size=(th, tw), mode='bilinear')
    nbb_list_s = []
    nbb_list_t = []
    for n in range(batch_size):
        tmp_list_s = []
        tmp_list_t = []
        dist = nnd_st[n, :, :, :].reshape(sh * sw)
        _, top_id = nnd_sd[n, :, :, :].topk(20, dim = 1, largest=True)
        for i in range(th):
            for j in range(tw):
                s_i, s_j = nnf_ts[n, :, i, j]
                t_i, t_j = nnf_st[n, :, s_i, s_j]
                if t_i == i and t_j == j:
                    gt_class = label[n, 0, s_i, s_j]
                    if gt_class != ignore_label and prob[n, gt_class, t_i, t_j] > 0.8:
                        tmp_list_s.append([s_i, s_j])
                        tmp_list_t.append([i, j])
                        count += 1
        pt_s = torch.Tensor(tmp_list_s)
        pt_t = torch.Tensor(tmp_list_t)
        nbb_list_s.append(tmp_list_s)
        nbb_list_t.append(tmp_list_t)
    return nbb_list_s, nbb_list_t
