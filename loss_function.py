import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine as cosine_distance

class LocalAggregationLoss(nn.Module):
    def __init__(self,
                 k_nearest_neighbours, number_of_centroids=5,
                 kmeans_n_init=1, clustering_repeats=10):
        super(LocalAggregationLoss, self).__init__()
        self.cluster = KMeans(n_clusters=number_of_centroids, n_init=clustering_repeats)
        self.normalizer = Normalizer()

    def forward(self, x, y):
        self.data = x
        x = x.type(torch.DoubleTensor)
        neu_x = self.normalizer.fit_transform(x.detach().numpy())
        y_hat = self.cluster.fit(neu_x).labels_
        # Compute the probability density for the codes given the constants of the memory bank
        v = F.normalize(x, p=2, dim=1)
        d1 = self._prob_density(v, background_neighbours)
        d2 = self._prob_density(v, neighbour_intersect)
        
        return torch.sum(torch.log(d1) - torch.log(d2)) / codes.shape[0]

    def bacward(self):
        self.data = self.data - 1/3



    