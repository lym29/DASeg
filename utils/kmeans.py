import torch

class KMEANS:
    def __init__(self, n_clusters=20, max_iter=None, verbose=False, device = torch.device("cpu")):

        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        self.dists = torch.zeros(x.shape[0], self.n_clusters).to(self.device)
        while True:
            self.nearest_center(x)
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        self.representative_sample()

    def nearest_center(self, x):
        dists = self.dists
        for i in range(self.n_clusters):
            self.dists[:, i] = torch.norm(x - self.centers[i, :], dim=1)
        self.labels = torch.argmin(dists, dim=1)
        
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0).to(self.device)], (0))
        self.centers = centers

    def representative_sample(self):
        self.representative_samples = torch.argmin(self.dists, (0))
        
    def predict(self, y):
        dist = torch.sum(torch.mul(y - self.centers, y - self.centers), (1))
        return torch.argmin(dist)


def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

