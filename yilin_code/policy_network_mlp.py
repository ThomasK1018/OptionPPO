import pandas as pd
import numpy as np
import os
import torch as T
import torch.distributions as D
from torch import optim
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim, lr=0.0001, dr=0.2, model_dir="./model", name="Policy"):
        super(PolicyNet, self).__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.lr = lr
        self.model_dir = model_dir
        self.name = name
        #self.checkpoint_file = os.path.join(self.model_dir, name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.checkpoint_file = f"{self.model_dir}/{name}"

        self.fcin = nn.Linear(state_dim, 512)
        # self.bn = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fcout = nn.Linear(512, act_dim*2) #for each dimension, we determine the mean and log(std) of the action (*2)
        self.fctest = nn.Linear(1, act_dim*2) #debug use
        self.drp = nn.Dropout(p=dr)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.double()#precision - double
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        x = state.double()#precision - double
        x = F.relu(self.fcin(x)) #no batch in simulation
        x = self.drp(x)
        x = F.relu(self.fc1(x))
        x = self.drp(x)
        x = F.relu(self.fc2(x))
        x = self.drp(x)
        x = self.fcout(x)

        return x

    def sample(self, state):
        x = self.forward(state)
        mu, lsig = x.split(self.act_dim)
        # print(lsig)
        assert not T.any(T.isinf(lsig)), f"{state} {lsig}"
        assert not T.any(T.isnan(lsig)), f"{state} {lsig}"
        # d = D.normal.Normal(mu,T.exp(lsig)) #log of std is trained instead so that the network output with full range, use exp to recover sig
        # d = D.normal.Normal(mu,0.1)#fix std for now
        d = D.normal.Normal(T.tanh(mu),T.sigmoid(lsig)+1e-12) #variance \in (0,1)
        sample = d.rsample()

        return sample, d.log_prob(sample).sum(axis=-1) #DONE: return logp also?

    def logp(self, state, act):
        x = self.forward(state)
        mu, lsig = x.split(self.act_dim,dim=-1)
        # print(mu, lsig)
        assert not T.any(T.isinf(lsig)), f"{state} {lsig}"
        assert not T.any(T.isnan(lsig)), f"{state} {lsig}"
        # d = D.normal.Normal(mu,T.exp(lsig))
        # d = D.normal.Normal(mu,0.1)
        d = D.normal.Normal(T.tanh(mu),T.sigmoid(lsig)+1e-12) #variance \in (0,1)

        return d.log_prob(act).sum(axis=-1), d.entropy()

    def evaluate(self, state):
        x = self.forward(state)
        mu, lsig = x.split(self.act_dim)
        return mu, -2.5066-lsig

    def save(self, suffix=""):
        print(f'... saving {self.name+suffix} ...')
        #print(self.checkpoint_file)
        #print(suffix)
        T.save(self.state_dict(), self.checkpoint_file+suffix)

    def load(self, suffix=""):
        print(f'... loading {self.name+suffix} ...')
        self.load_state_dict(T.load(self.checkpoint_file+suffix))

