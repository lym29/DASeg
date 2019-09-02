import numpy as np
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
    def __init__(self, a, b, patch_size=3):
        super(PatchMatch, self).__init__()
        self.source_map = a
        self.target_map = b
        self.f_dim = self.source_map.shape[2]
        self.patch_size = patch_size

        self.nnf = np.zeros((a.shape[0], a.shape[1], 2)).astype(np.int)  # The NNF
        self.nnd = np.zeros(a.shape)  # The NNF distance map

    def build_feat_for_flann(self, patch_center, feature_map):
        map_2d_size = np.asarray(feature_map.shape[:2])

        patch_min, patch_max = get_patch_minmax(map_2d_size, patch_center, self.patch_size)

        patch = feature_map[patch_min[0]:patch_max[0], patch_min[1]:patch_max[1], :]
        patch_array = patch.reshape((-1, self.f_dim))
        output = np.zeros((self.patch_size * self.patch_size, self.f_dim))

        output[:patch_array.shape[0], :] = patch_array

        return output.reshape((1, -1))

    def build_dataset_for_flann(self, feature_map):
        dataset_num = feature_map.shape[0] * feature_map.shape[1]
        dataset = np.zeros((dataset_num, self.patch_size * self.patch_size * self.f_dim))
        count = 0
        for i in range(feature_map.shape[0]):
            for j in range(feature_map.shape[1]):
                feat = self.build_feat_for_flann(np.array([i, j]), feature_map)
                dataset[count, :] = feat
                count += 1
        return dataset

    def find_nnf(self):
        flann = pyflann.FLANN()
        pts = self.build_dataset_for_flann(self.target_map)
        q_pts = self.build_dataset_for_flann(self.source_map)
        result_id, dists = flann.nn(pts, q_pts, 1, algorithm='kdtree', trees=4)
        count = 0
        for i in range(self.source_map.shape[0]):
            for j in range(self.source_map.shape[1]):
                id_in_targ = result_id[count]
                idy, idx = id_in_targ // self.target_map.shape[1], id_in_targ % self.target_map.shape[1]
                self.nnf[i, j, :] = np.array([idy, idx])
                self.nnd[i, j, :] = dists[count]
                count += 1

    def forward(self):
        self.find_nnf()
        return self.nnf


def bds_vote(ref, nnf_sr, nnf_rs, ignore_label = 255, patch_size=3):
    """
    Reconstructs an image or feature map by bidirectionaly
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
            if abs(ref[ry, rx]-255) < 1:
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
            if abs(ref[ry, rx]-255) < 1:
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
    guide[total_weight < 1e-8] = 255
    return guide


