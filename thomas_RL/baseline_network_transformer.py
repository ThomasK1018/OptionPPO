import os #PS: explain os module?
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import optim 
from transformer_network import TransformerNN
class BaselineNet(nn.Module):
    def __init__(self, input_dim, state_dim, num_heads = 4, intermediate_dim = 256, lr=0.0001, dr=0.2, model_dir="./model", name="Baseline"):
        super(BaselineNet, self).__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = 1
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.lr = lr
        self.dr = dr
        self.model_dir = model_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.model_dir, name)

        self.layer = TransformerNN(input_dim = self.input_dim, state_dim = self.state_dim, output_dim = self.output_dim, num_heads = self.num_heads, intermediate_dim = self.intermediate_dim, lr = self.lr, dr = self.dr)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.layer(x)
        return x

    def save(self, suffix=""):
        print(f'... saving {self.name+suffix} ...')
        torch.save(self.state_dict(), self.checkpoint_file+suffix)

    def load(self, suffix=""):
        print(f'... loading {self.name+suffix} ...')
        self.load_state_dict(torch.load(self.checkpoint_file+suffix))
        
#state_dim = 20
#input = torch.tensor([1, 3, 5], dtype = torch.float32)
#input_dim = len(input)
#output_dim = 6
#num_heads = 2
#model = BaselineNet(input_dim = input_dim, state_dim = state_dim, output_dim = output_dim, num_heads = num_heads)

#print(model(input))