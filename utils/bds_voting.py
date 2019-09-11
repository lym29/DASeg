import numpy as np
import torch
import pyflann
import torch.nn as nn
import torch.functional as F


def build_extended_map(self, x):
    n, c, h, w = x.size()
    step = self.patch_size // 2
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

    ref = F.upsample(ref, size=(sh, sw), mode='bilinear')

    num_s = sh * sw
    num_t = th * tw

    ex_ref = build_extended_map(ref)
    ex_guide = torch.zeros(batch_size, c, th + 2 * step, tw + 2 * step)
    ex_weight = build_extended_map(torch.zeros(batch_size, c, th + 2 * step, tw + 2 * step))

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
    n, h, w = labels.size()
    p = torch.zeros(n, num_classes, h, w)
    for c in range(num_classes):
        p[:, c, :, :][labels == c] = 1
    return p


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
        y = self.build_extended_map(x)
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
            pts = pts_tensor[n, :, :].numpy().transpose(1, 0)
            q_pts = q_pts_tensor[n, :, :].numpy().transpose(1, 0)
            result_id, dists = flann.nn(pts, q_pts, 1, algorithm='kdtree', trees=4)
            idy, idx = result_id // tw, result_id % tw
            self.nnf[n, 0, :, :] = torch.from_numpy(idy.reshape(th, tw))
            self.nnf[n, 1, :, :] = torch.from_numpy(idx.reshape(th, tw))
            self.nnd[n, 0, :, :] = torch.from_numpy(dists.reshape(th, tw))

    def forward(self):
        self.find_nnf()
        return self.nnf


