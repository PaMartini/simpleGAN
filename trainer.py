
from torch.utils.data import Dataset
import torch
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt


class Trainer:

    def __init__(self,
                 model,                        # Model: {'Generator': gen, 'Discriminator': dis}
                 crit,                         # Loss function
                 optim=None,                   # Optimizer: {'GenOpt': genopt, 'DisOpt': disopt}
                 train_dl=None,                # Data loader training
                 cuda=True):                   # Whether to use the GPU

        self._G = model['Generator']
        self._D = model['Discriminator']
        self._crit = crit
        self._optim_G = optim['GenOpt']
        self._optim_D = optim['DisOpt']
        self._train_dl = train_dl
        self._cuda = cuda

        if cuda:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._gen = self._G.to(device)
            self._dis = self._D.to(device)
            self._crit = crit.to(device)

        def trainD():
            pass

        def trainG():
            pass

