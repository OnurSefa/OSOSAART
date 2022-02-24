import torch
from torch import nn
import matplotlib.pyplot as plt

import numpy as np
import math
from torch import nn


class OsosaLoss(nn.Module):
    def __init__(self, y_hat, number_of_centroids=5, radius=1):
        super(OsosaLoss, self).__init__()
        self.centers = self.find_points(number_of_centroids, y_hat, radius)

    def forward(self, x):
        # distances = torch.cdist(x.float(), self.centers.float(), p=2)
        # difference = x.float() - self.centers.float()
        # squared_differences = torch.mul(difference, difference)
        # total_squared_differences = torch.sum(squared_differences, dim=1)
        # distances = torch.sqrt(total_squared_differences)
        # total_distance = torch.sum(distances)
        # avg_distance = torch.div(total_distance, self.centers.shape[0])
        loss = nn.MSELoss(reduction='mean')
        return loss(x, self.centers.float())

    @staticmethod
    def find_points(center_count, y_hat, radius=1):
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
        correct_points = []
        for y in y_hat:
            current_correct_point = points[int(y)]
            correct_points.append(current_correct_point)

        correct_points = torch.tensor(correct_points, requires_grad=False)
        return correct_points

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
