import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import clip
from matplotlib.image import imread

device = "cuda" if torch.cuda.is_available() else 'cpu'


class PixelModel(nn.Module):
    def __init__(self):
        super(PixelModel, self).__init__()
        self.sequential = nn.Sequential(nn.Linear(300, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 4096),
                                        nn.ReLU(),
                                        nn.Linear(4096, 12288))

    def forward(self, x_given):
        return self.sequential(x_given)


def show_image(image_numpy):
    plt.imshow(image_numpy)
    plt.show()


def create_model(given_words, l_rate=0.003, e=200):
    words = np.random.uniform(0, 1, (10, 300))
    model = PixelModel().to(device)

    X = torch.from_numpy(words).float().to(device)
    Y = torch.from_numpy(words).float().to(device)


if __name__ == '__main__':
    random_image = np.random.randint(0, 256, (64, 64, 3))
    show_image(random_image)



