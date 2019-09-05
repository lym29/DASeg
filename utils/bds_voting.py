import numpy as np
import torch
import pyflann
import torch.nn as nn


def get_patch_minmax(map_2d_size, patch_center, patch_size):
    patch_min = patch_center - patch_size // 2
    patch_min[patch_min < 0] = 0
    patch_max = patch_center + patch_size // 2 + 1
    patch_max[patch_max > map_2d_size] = map_2d_size[patch_max > map_2d_size]
    return patch_min, patch_max


def modify_patch_minmax(sy, sx, ry, rx, s_patch_min, s_patch_max, r_patch_min, r_patch_max):
    s_min_dy, s_min_dx, s_max_dy, s_max_dx = sy - s_patch_min[0], sx - s_patch_min[1], s_patch_max[0] - sy, s_patch_max[
        1] - sx
    r_min_dy, r_min_dx, r_max_dy, r_max_dx = ry - r_patch_min[0], rx - r_patch_min[1], r_patch_max[0] - ry, r_patch_max[
        1] - rx
    neg_dy, neg_dx = min(s_min_dy, r_min_dy), min(s_min_dx, r_min_dx)
    pos_dy, pos_dx = min(s_max_dy, r_max_dy), min(s_max_dx, r_max_dx)

    out_r_patch_min, out_r_patch_max = np.array([ry - neg_dy, rx - neg_dx]), np.array([ry + pos_dy, rx + pos_dx])
    out_s_patch_min, out_s_patch_max = np.array([sy - neg_dy, sx - neg_dx]), np.array([sy + pos_dy, sx + pos_dx])
    return out_s_patch_min, out_s_patch_max, out_r_patch_min, out_r_patch_max


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
        step = patch_size // 2
        y = torch.zeros((n, c, h + 2 * step, w + 2 * step))
        y[:, :, step:step + h, step:step + w] = x
        y[:, :, step:step + h, :step] = x[:, :, :, 0:1].expand((n, c, h, step))
        y[:, :, step:step + h, step + w:step + 2 * w] = x[:, :, :, -1:w + 2 * step].expand((n, c, h, step))
        y[:, :, :step, :] = y[:, :, step:step + 1, :].expand((n, c, step, w + 2 * step))
        y[:, :, h + step:, :] = y[:, :, h + step - 1:h + step, :].expand((n, c, step, w + 2 * step))

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
        flann = pyflann.FLANN()
        pts_tensor = self.build_pts_set_for_flann(self.target_map)
        q_pts_tensor = self.build_pts_set_for_flann(self.source_map)
        for n in range(self.batch_size):
            pts = pts_tensor[n, :, :].numpy().transpose(1, 0)
            q_pts = q_pts_tensor[n, :, :].numpy().transpose(1, 0)
            result_id, dists = flann.nn(pts, q_pts, 1, algorithm='kdtree', trees=4)
            count = 0
            for i in range(self.source_map.shape[0]):
                for j in range(self.source_map.shape[1]):
                    id_in_targ = result_id[count]
                    idy, idx = id_in_targ // self.target_map.shape[1], id_in_targ % self.target_map.shape[1]
                    self.nnf[n, i, j, :] = np.array([idy, idx])
                    self.nnd[n, i, j, :] = dists[count]
                    count += 1

    def forward(self):
        self.find_nnf()
        return self.nnf


def bds_vote(ref, nnf_sr, nnf_rs, ignore_label=255, patch_size=3):
    """
    Reconstructs an image or feature map by bidirectional
    similarity voting
    """

    src_height = nnf_sr.shape[0]
    src_width = nnf_sr.shape[1]
    ref_height = nnf_rs.shape[0]
    ref_width = nnf_rs.shape[1]
    channel = ref.shape[2]

    guide = np.zeros((src_height, src_width, channel))
    ws = 1 / (src_height * src_width)
    wr = 1 / (ref_height * ref_width)

    total_weight = np.zeros((src_height, src_width, channel))

    # coherence
    # The S->R forward NNF enforces coherence
    for sy in range(src_height):
        for sx in range(src_width):
            ry, rx = nnf_sr[sy, sx]
            if abs(ref[ry, rx]-ignore_label) < 1:
                continue
            r_patch_min, r_patch_max = get_patch_minmax(np.asarray(nnf_rs.shape[:2]), np.array([ry, rx]), patch_size)
            s_patch_min, s_patch_max = get_patch_minmax(np.asarray(nnf_sr.shape[:2]), np.array([sy, sx]), patch_size)

            rpatch_size, spatch_size = r_patch_max - r_patch_min, s_patch_max - s_patch_min
            if not ((rpatch_size == np.array([patch_size, patch_size])).all() and (spatch_size == np.array([patch_size, patch_size])).all()):
                s_patch_min, s_patch_max, r_patch_min, r_patch_max = \
                     modify_patch_minmax(sy, sx, ry, rx, s_patch_min, s_patch_max, r_patch_min, r_patch_max)

            guide[s_patch_min[0]:s_patch_max[0], s_patch_min[1]:s_patch_max[1], :] += \
                ws * ref[r_patch_min[0]:r_patch_max[0], r_patch_min[1]:r_patch_max[1], :]
            total_weight[s_patch_min[0]:s_patch_max[0], s_patch_min[1]:s_patch_max[1], :] += ws

    # completeness
    # The R->S backward NNF enforces completeness
    for ry in range(ref_height):
        for rx in range(ref_width):
            if abs(ref[ry, rx]-ignore_label) < 1:
                continue
            sy, sx = nnf_rs[ry, rx]
            r_patch_min, r_patch_max = get_patch_minmax(np.asarray(nnf_rs.shape[:2]), np.array([ry, rx]), patch_size)
            s_patch_min, s_patch_max = get_patch_minmax(np.asarray(nnf_sr.shape[:2]), np.array([sy, sx]), patch_size)

            rpatch_size, spatch_size = r_patch_max - r_patch_min, s_patch_max - s_patch_min
            if not ((rpatch_size == np.array([patch_size, patch_size])).all() and (spatch_size == np.array([patch_size, patch_size])).all()):
                s_patch_min, s_patch_max, r_patch_min, r_patch_max = \
                    modify_patch_minmax(sy, sx, ry, rx, s_patch_min, s_patch_max, r_patch_min, r_patch_max)

            guide[s_patch_min[0]:s_patch_max[0], s_patch_min[1]:s_patch_max[1], :] += \
                wr * ref[r_patch_min[0]:r_patch_max[0], r_patch_min[1]:r_patch_max[1], :]
            total_weight[s_patch_min[0]:s_patch_max[0], s_patch_min[1]:s_patch_max[1], :] += wr

    total_weight[total_weight < 1e-8] = 1
    guide /= total_weight
    guide[total_weight < 1e-8] = ignore_label
    return guide


