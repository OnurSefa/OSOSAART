
import io
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import handler
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans


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
    loss_f = loss_function
    loss_fn = nn.CrossEntropyLoss()
    y = y.long()

    # TODO batch halinde alabiliriz daha cok data kullandigimizda
    for e in range(epoch):
        prediction = model(x)
        loss = loss_f(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 10 == 0:
            print("Loss in e={} is {}".format(e, loss))
        

if __name__ == '__main__':
    data, labels = handler.take_data()
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    data = data.float()
    labels = labels.float()
    train(data, labels)
    print('b')
