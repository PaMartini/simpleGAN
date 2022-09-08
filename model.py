
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import numpy as np


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super().__init__()
        self.g_id = g_input_dim
        self.g_od = g_output_dim

        self.layers = nn.Sequential(
            nn.Linear(self.g_id, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, self.g_od),
        )

    def forward(self, input_tensor):
        return torch.sigmoid(self.layers(input_tensor))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super().__init__()
        self.d_id = d_input_dim

        self.layers = nn.Sequential(
            nn.Linear(self.d_id, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_tensor):
        return torch.sigmoid(self.layers(input_tensor))
