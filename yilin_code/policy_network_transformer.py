import pandas as pd
import numpy as np
import os
import torch
import torch.distributions as D
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from transformer_network import TransformerNN
from torch import optim

class PolicyNet(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim, num_heads = 4, intermediate_dim = 256, lr=0.0001, dr=0.2, model_dir="./model", name="Policy"):
        super(PolicyNet, self).__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.lr = lr
        self.dr = dr
        self.model_dir = model_dir
        self.name = name
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.checkpoint_file = f"{self.model_dir}/{name}"
        
        self.layer = TransformerNN(input_dim = self.input_dim, state_dim = self.state_dim, output_dim = self.output_dim * 2, num_heads = self.num_heads, intermediate_dim = self.intermediate_dim, lr = self.lr, dr = self.dr)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.double()#precision - double
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, x):
        x = self.layer(x)

        return x


    def sample(self, state):
        x = self.forward(state)
        mu, lsig = x.split(self.output_dim, dim=-1)
        d = D.normal.Normal(torch.tanh(mu), torch.sigmoid(lsig) + 1e-12)  # Variance ∈ (0, 1)
        sample = d.rsample()
        return sample, d.log_prob(sample).sum(axis=-1)

    def logp(self, state, act):
        x = self.forward(state)
        #print(x)
        mu, lsig = x.split(self.output_dim, dim=-1)
        d = D.normal.Normal(torch.tanh(mu), torch.sigmoid(lsig) + 1e-12)  # Variance ∈ (0, 1)
        #print(d)
        #print((act))
        return d.log_prob(act).sum(axis=-1), d.entropy()

    def evaluate(self, state):
        x = self.forward(state)
        mu, lsig = x.split(self.output_dim, dim=-1)
        return mu, -2.5066 - lsig

    def save(self, suffix=""):
        print(f'... saving {self.name + suffix} ...')
        torch.save(self.state_dict(), self.checkpoint_file + suffix)

    def load(self, suffix=""):
        print(f'... loading {self.name + suffix} ...')
        self.load_state_dict(torch.load(self.checkpoint_file + suffix))
        
#state_dim = 20
#input = torch.tensor([1, 3, 5], dtype = torch.float32)
#input_dim = len(input)
#output_dim = 1
#num_heads = 2
#model = PolicyNet(input_dim = input_dim, state_dim = state_dim, output_dim = output_dim, num_heads = num_heads)

#print(model(input))