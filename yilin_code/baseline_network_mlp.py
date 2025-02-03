import os #PS: explain os module?
import numpy as np
import pandas as pd
import torch as T  #PS: explain or reference online resources for pytorch
import torch.nn as nn
import torch.nn.functional as F  #PS: this may be advanced concept, explain or footnote
import torch.distributions as D
from torch import optim  #PS: what is the optimization algorithm in pytorch?  Any description needed?

class BaselineNet(nn.Module):
    def __init__(self, state_dim, lr=0.0001, dr=0.2, model_dir="./model", name="Baseline"):
        super(BaselineNet, self).__init__()

        self.state_dim = state_dim
        self.lr = lr
        self.model_dir = model_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.model_dir, name)

        self.fcin = nn.Linear(state_dim, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.bn3 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fcout = nn.Linear(512, 1) #for each dimension, we determine the mean and log(std) of the action (*2)
        self.drp = nn.Dropout(p=dr)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = state.double()
        x = F.relu(self.fcin(x))
        x = self.drp(x)
        x = F.relu(self.fc1(x))
        x = self.drp(x)
        # x = self.bn3(F.relu(self.fc2(x)))
        # x = self.drp(x)
        x = self.fcout(x)
        return x

    def save(self, suffix=""):
        print(f'... saving {self.name+suffix} ...')
        T.save(self.state_dict(), self.checkpoint_file+suffix)

    def load(self, suffix=""):
        print(f'... loading {self.name+suffix} ...')
        self.load_state_dict(T.load(self.checkpoint_file+suffix))