import torch
from torch import nn
import matplotlib.pyplot as plt

import numpy as np
import math


class OsosaLoss(nn.Module):
    def __init__(self, number_of_centroids=5, radius=1):
        super(OsosaLoss, self).__init__()
        self.centers = self.find_points(number_of_centroids, radius)

    def forward(self, x, y):
        y_hat = np.zeros((y.shape[0], self.centers.size(dim=0)))
        indices = np.arange(y.shape[0])
        y_hat[indices, y[indices]] = 1
        y_hat = torch.tensor(y_hat, requires_grad=False).long()
        distances = torch.cdist(x.float(), self.centers.float(), p=2)
        related = torch.mul(y_hat, distances)
        total_distance = torch.sum(related)
        avg_distance = torch.div(total_distance, y.shape[0])
        return avg_distance

    def find_points(self, center_count, radius=1):
        points = []
        degree_increment = 2 * math.pi / center_count
        cos_degree = math.pi / 2
        sin_degree = math.pi / 2
        for i in range(center_count):
            x_value = radius * math.cos(cos_degree)
            y_value = radius * math.sin(sin_degree)
            points.append((x_value, y_value))

            cos_degree += degree_increment
            sin_degree += degree_increment

        points = np.array(points)
        points = torch.tensor(points, requires_grad=False)
        return points

    def show(self, x, y):
        x_data = x.clone().detach().numpy()
        y_data = y.clone().detach().numpy()
        plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
        plt.show()


if __name__ == '__main__':
    loss_fn = OsosaLoss(number_of_centroids=4)
    x_data = [[0, 1], [0, 1], [1, 0], [1, 0]]
    x_data = np.array(x_data)
    x_data = torch.tensor(x_data)
    y_data = [0, 0, 3, 3]
    y_data = np.array(y_data)
    loss = loss_fn(x_data, y_data)
    # loss_fn.show(x_data, y_data)
    print(loss)
