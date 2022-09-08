from torch.utils.data import Dataset
import torch
import torchvision as tv
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model import Generator, Discriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = tv.transforms.Compose([
    tv.transforms.ToTensor()
    #tv.transforms.Normalize(mean=0.5, std=0.5)   # no normalization data is in [0, 1] already
])

train_data = tv.datasets.MNIST(root='./train_data', train=True, download=True, transform=transform)

bs = 10
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=bs, shuffle=True)

'''# plot data:
examples = enumerate(train_loader)
el = next(examples)
print(el[0])
print(el[1][0].max())  # batch x 1 x 28 x 28
print(el[1][1])
batch_idx, (example_data, example_targets) = next(examples)
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("label: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()'''

print(train_data.data.size()[1])

#gen = Generator()

criterion = nn.BCELoss()

# todo: train

