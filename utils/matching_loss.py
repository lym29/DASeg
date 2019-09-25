import torch
import torch.nn.functional as F
from utils import kmeans


def build_prob_map(labels, num_classes):
    n, _, h, w = labels.size()
    l = labels.squeeze(1)
    p = torch.zeros(n, num_classes, h, w)
    for c in range(num_classes):
        p[:, c, :, :][l == c] = 1
    return p


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


def compute_matching_loss(p1, p2, f1, f2, softmax):
    """
    p2 and f2 are guidance
    """
    p1 = softmax(p1)
    p2 = softmax(p2)
    weight = torch.norm(f1 - f2, dim=1)
    weight = 1 - torch.pow(weight, 2)/2
    loss = torch.norm(p1 - p2, dim=1)
    loss = weight * loss
    return torch.mean(loss)


def gaussian_mask(mu, sigma, d, norm_num, grid_num):
    """
    mu: Tensor, centre of the first Gaussian.
    sigma: Tensor, standard deviation of Gaussians.
    d: Tensor, shift between Gaussian centres.
    """
    b_size, c, _= mu.size()
    normal_center = mu + d * torch.arange(0, norm_num).repeat(b_size, c, 1).float()
    grid_center = torch.arange(0, grid_num).repeat(b_size, c, 1).float()
    mask = torch.exp(-0.5*((grid_center[:,:,:,None] - normal_center[:,:,None,:])/sigma).pow(2))
    normalized_mask = mask / (mask.sum(dim = 2, keepdim = True)+1e-8)
    return normalized_mask


def gaussian_glimpse(img_tensor, transform_params, crop_size):
    """
    :param img_tensor: Tensor of size (batch_size, channels, Height, Width)
    :param transform_params: Tensor of size (batch_size, 6), where params are (mean_y, std_y, d_y, mean_x, std_x, d_x) specified in pixels.
    :param crop_size: tuple of 2 ints, size of the resulting crop
    """
    # parse arguments
    h, w = crop_size
    H, W = img_tensor.size()[2:]
    uy, sy, dy, ux, sx, dx = transform_params
    # create Gaussian masks, one for each axis
    Ay = gaussian_mask(uy, sy, dy, h, H)
    Ax = gaussian_mask(ux, sx, dx, w, W)
    # extract glimpse
    glimpse = torch.matmul(torch.matmul(Ay.transpose(3,2), img_tensor), Ax)
    return glimpse


def find_NBB(nnf_st, nnf_ts, label, prob, ignore_label = 255):
    batch_size, _, sh, sw = nnf_st.size()
    _, _, sh, sw = nnf_st.size()
    _, _, th, tw = nnf_ts.size()

    label = F.upsample(label.float(), size=(sh, sw), mode='nearest')
    label = label.long()
    nbb_list_s = []
    nbb_list_t = []
    for n in range(batch_size):
        tmp_list_s = []
        tmp_list_t = []
        for i in range(th):
            for j in range(tw):
                s_i, s_j = nnf_ts[n, :, i, j]
                t_i, t_j = nnf_st[n, :, s_i, s_j]
                if t_i == i and t_j == j:
                    gt_class = label[n, 0, s_i, s_j]
                    if gt_class != ignore_label and prob[n, gt_class, t_i, t_j] > 0.5:
                        tmp_list_s.append([s_i, s_j])
                        tmp_list_t.append([i, j])
        nbb_list_s.append(tmp_list_s)
        nbb_list_t.append(tmp_list_t)
    return nbb_list_s, nbb_list_t


def get_crop_size(pred, nbb_list, scale, cuda_device):
    crop_size_list = []
    b_size, c, h, w = pred.size()
    for n in range(b_size):
        tmp = []
        data = torch.zeros(h, w, c+2)
        data[:, :, :c] = pred[n, :, :, :].detach().permute(1, 2, 0)
        data[:, :, c] = torch.arange(h).repeat(w, 1).reshape(h, w) / h   # pos i
        data[:, :, c+1] = torch.arange(w).reshape(1, w).repeat(h, 1) / w # pos j
        
        clus_num = 2*len(nbb_list[n])
        if clus_num == 0:
            crop_size_list.append(tmp)
            continue
        
        estimator = kmeans.KMEANS(n_clusters=clus_num, max_iter = 100, device=cuda_device)
        
        init_points = torch.zeros(len(nbb_list[n]), c+2)
        for k in range(len(nbb_list[n])):
            pt = nbb_list[n][k]
            i, j = int(pt[0]*scale[0]), int(pt[1]*scale[1])
            init_points[k, :] = data[i, j, :]
        
        estimator.fit(data.reshape(h*w, c+2).to(cuda_device), init_points)
        kmeans_labels = estimator.labels.reshape(h, w)
        for pt in nbb_list[n]:
            i, j = int(pt[0]*scale[0]), int(pt[1]*scale[1])
            pt_class = kmeans_labels[i, j]
            far_dist = data[:,:,c:].norm(dim=2)
            far_dist[kmeans_labels != pt_class] = 0
            far_pos = torch.argmax(far_dist)
            crop_size = [4 + abs(far_pos // w - i), 4 + abs(far_pos % w - j)]
            tmp.append(crop_size)
        crop_size_list.append(tmp)
    return crop_size_list


def crop_img(pred, scale, anchor, crop_size):
    _, _, h, w = pred.size()
    center = anchor.long() * scale.long()
    crop_size = torch.Tensor([crop_size[0], crop_size[1]]).long()
    img_size = torch.Tensor([h, w]).long()
    p_min = center - crop_size
    p_max = center + crop_size
    p_min[p_min < 0] = 0
    p_max[p_max >= img_size] = img_size[p_max >= img_size]
    return pred[:, :, p_min[0]:p_max[0], p_min[1]:p_max[1]]


