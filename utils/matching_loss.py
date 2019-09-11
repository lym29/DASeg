import torch


def compute_matching_loss(p1, p2, f1, f2):
    weight = torch.norm(f1 - f2, dim=1)
    weight = 1 - torch.pow(weight, 2)/2
    loss = torch.norm(p1 - p2, dim=1)
    loss = weight * loss
    return torch.mean(loss)
