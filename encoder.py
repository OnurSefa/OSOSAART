
import io
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import handler
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
import loss_function

class Ososa(nn.Module):
    def __init__(self):
        super(Ososa, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (10, 10))
        self.conv2 = nn.Conv2d(6, 4, (8, 8))
        self.conv3 = nn.Conv2d(4, 2, (6, 6))
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d((3, 3))
        self.pool2 = nn.MaxPool2d((3, 3))
        self.pool3 = nn.MaxPool2d((5, 5))
        self.linear1 = nn.Linear(2 * 5 * 5, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x


def train(x, y, epoch=200, learning_rate=0.005):
    model = Ososa()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # loss_f = loss_function
    # loss_fn = nn.CrossEntropyLoss()
    values = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1], [-1, 0], [-1, 0], [-1, 0]]).float()
    x = x[0:7, :, :, :]
    y = y.long()
    # loss_f = loss_function.OsosaLoss(y, number_of_centroids=5)
    loss_f = nn.MSELoss()
    y_hat = loss_function.OsosaLoss.find_points(5, y, 1).float()
    y_hat.requires_grad = False
    

    # TODO batch halinde alabiliriz daha cok data kullandigimizda
    for e in range(epoch):
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_f(prediction, values)
        loss.backward()
        optimizer.step()

        if e % 10 == 0:
            print("Loss in e={} is {}".format(e, loss))
           #  loss_f.show(prediction, y)


if __name__ == '__main__':
    data, labels = handler.take_data()
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    data = data.float()
    labels = labels.float()
    train(data, labels)
    print('b')
