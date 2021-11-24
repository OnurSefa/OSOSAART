import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import clip
from PIL import Image
import torch.optim as optim
from torch.nn import functional as F


device = "cuda" if torch.cuda.is_available() else 'cpu'


class PixelModel(nn.Module):
    def __init__(self):
        super(PixelModel, self).__init__()
        self.sequential = nn.Sequential(nn.Linear(77, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 4096),
                                        nn.ReLU(),
                                        nn.Linear(4096, 12288))

    def forward(self, x_given):
        return self.sequential(x_given)


def show_image(image_numpy):
    plt.imshow(image_numpy)
    plt.show()


def one_sided_clip_loss(input, target, labels=None, logit_scale=100):
    input_normed = F.normalize(input, dim=-1)
    target_normed = F.normalize(target, dim=-1)
    logits = input_normed @ target_normed.T * logit_scale
    if labels is None:
        labels = torch.arange(len(input), device=logits.device)
    return F.cross_entropy(logits, labels)


def create_model(dataset_function, l_rate=0.003, epoch=200, data_length=10000, dataset_file='dataset.txt'):

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    words = dataset_function(dataset_file)
    words = words[:data_length]
    tokenized_words = clip.tokenize(words)
    model = PixelModel().to(device)

    loss_fn = clip_model
    optimizer = optim.Adam(model.parameters(), lr=l_rate)

    X = torch.from_numpy(tokenized_words.numpy()).float().to(device)

    tokenized_words = tokenized_words.reshape(-1, data_length).to(device)

    train_losses = []
    for e in range(epoch):
        model.train()

        guessed = model(X)

        loss = one_sided_clip_loss(guessed.reshape(12288, -1).type(torch.float32), tokenized_words.type(torch.float32))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if e % 10 == 0:
            print('epoch:', e)
            train_losses.append(loss)

    return model.state_dict(), train_losses


def take_dataset(dataset_file='dataset.txt'):
    all_words = []
    with open(dataset_file, 'r') as read_file:
        words = read_file.readlines()
        for word in words:
            text = word[:-1]
            all_words.append(text)
    return all_words


if __name__ == '__main__':

    # random_image = np.random.randint(0, 256, (64, 64, 3))
    # random_image = Image.fromarray(random_image.astype('uint8'), 'RGB')

    states, losses = create_model(take_dataset)
    plt.plot(losses)
    plt.show()





