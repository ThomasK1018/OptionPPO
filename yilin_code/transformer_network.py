import torch
import torch.nn as nn
import numpy as np
from transformer_blocks import PositionalEncoder
from transformer_blocks import TransformerEncoderBlock
from transformer_blocks import AttentionPooling
from torch import optim
import os

class TransformerNN(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim, num_heads, intermediate_dim = 256, lr=0.0001, dr=0.2, model_dir="./model", name="Baseline"):
        super(TransformerNN, self).__init__()

        self.state_dim = state_dim
        self.lr = lr
        self.model_dir = model_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.model_dir, name)
        
        self.fcin = nn.Linear(input_dim, state_dim)
        self.pe = PositionalEncoder(state_dim)
        self.encoder = TransformerEncoderBlock(state_dim, num_heads, intermediate_dim, dropout = dr)
        self.attpool = AttentionPooling(state_dim)
        self.fcout = nn.Linear(state_dim, output_dim)
        self.drp = nn.Dropout(p = dr)
        #x1 = nn.Linear(len(input), i)
        #x2 = PositionalEncoder(i)
        #x3 = TransformerEncoderBlock(i, 2, 16)
        #x4 = AttentionPooling(i)
        #x5 = nn.Linear(i, 1)
        

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        #x = x.double()
        reshape = True if len(x.shape) == 1 else False
        if reshape:
            x = x.reshape(1, len(x))
        x = self.fcin(x)
        x = self.pe(x)
        x = self.encoder(x)
        x = self.attpool(x)
        x = self.fcout(x)
        if reshape:
            x = x.squeeze(1)
        return x

    def save(self, suffix=""):
        print(f'... saving {self.name+suffix} ...')
        torch.save(self.state_dict(), self.checkpoint_file+suffix)

    def load(self, suffix=""):
        print(f'... loading {self.name+suffix} ...')
        self.load_state_dict(torch.load(self.checkpoint_file+suffix))


#state_dim = 10
#input = torch.tensor([[1, 3, 5]], dtype = torch.float32)
#if len(input.shape) == 1:
#    input_dim = len(input)
#else:
#    input_dim = input.shape[1]
#print(input_dim)
#output_dim = 3
#num_heads = 2
#model = TransformerNN(input_dim = input_dim, state_dim = state_dim, output_dim = output_dim, num_heads = num_heads)

#print(model(input))


