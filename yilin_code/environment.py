import os #PS: explain os module?
from typing import Tuple
from typing import Type
from typing import Optional
import numpy as np
import pandas as pd
import torch as T  #PS: explain or reference online resources for pytorch
import torch.nn as nn
import torch.nn.functional as F  #PS: this may be advanced concept, explain or footnote
import torch.distributions as D
from torch import optim  #PS: what is the optimization algorithm in pytorch?  Any description needed?
import matplotlib.pyplot as plt
import copy  #PS: OK I don't know this one.  Will research
from reinforcement import Reinforcement
from heston import Heston

class Environment(Reinforcement, nn.Module):
    def __init__(self, eps=0.2, lr=0.0001, dr=0.2, gamma=0.99, lam=0.95, entrpy_scaler=0.0001):

        state_dim = 4 #change this when incorporating more features
        act_dim = 1

        self.entrpy_scaler = entrpy_scaler

        super(Environment, self).__init__(state_dim, act_dim, eps=eps, lr=lr, dr=dr, gamma=gamma, lam=lam)

        # draw from a joint distribution in the future
        self.r = 0.007                                           # drift
        rho = 0.3                                         # correlation coefficient
        kappa = 30                                          # mean reversion coefficient
        theta = 0.13                                        # long-term mean of the variance
        sigma = 1.6                                       # (Vol of Vol) - Volatility of instantaneous variance
        
        self.Tunit = 1/252
        self.dt = self.Tunit/10
        self.model = Heston(self.r, kappa, theta, sigma, rho)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def reset(self):
        self.iat = 0
        self.v0 = 0.2                                          # spot variance
        self.S0 = 450                                            # spot stock price
        self.T = 20                                             # Terminal time
        self.K = 450	                                            # Stike

        self.option = self.model.price_put(self.S0, self.v0, self.K, self.T * self.Tunit, self.r)
        self.cash = self.option
        self.delta = 0
        
        self.w0 = self.cash

        self.moneyness = self.S0 / self.K - 1

        self.state = T.tensor([self.moneyness, np.sqrt(self.T * self.Tunit), self.delta, -0.5]).to(self.device)
        return

    def evolve(self, action) -> Tuple[T.Tensor, T.Tensor, bool]: #play with the data first
        oldstate = self.state.clone().detach()
        self.cash *= np.exp(self.r)
        self.cash -= self.S0 * (action[0].item() + np.abs(action[0].item() * 0.01))
        self.delta += action[0].item()
        self.iat += 1
        
        temp = [self.S0, self.option]
        #print(self.S0)
        #print(self.v0)
        #print(self.Tunit)
        #print(self.dt)
        self.S0, self.v0 = self.model.simulate_n(self.S0, self.v0, self.Tunit, self.dt)

        self.option = self.model.price_put(self.S0, self.v0, self.K, (self.T - self.iat) * self.Tunit, self.r)
        
        temp[0] = self.S0 - temp[0]
        temp[1] = self.option - temp[1]
        dd = temp[1] / temp[0]
        
        reward = T.minimum((T.tensor(self.cash + self.S0 * self.delta - self.option / self.w0) - 1.).to(self.device), T.zeros(1,dtype=T.double).to(self.device))
        # print(reward)

        done = False
        if (self.iat == self.T) or (reward.item() < -2):
            done = True
        else:
            self.moneyness = self.S0 / self.K - 1

            self.state = T.tensor([self.moneyness, np.sqrt((self.T - self.iat) * self.Tunit), self.delta, dd]).to(self.device)

        return oldstate, reward, done


    def objective(self, batch: dict) -> T.tensor:
        """
        for each epoch/call, detach the original logp; re-evaluate the new logp for exploitation ratio. evaluate the advantage for the updated network.
        modify the clip(pi/pi_k,1-eps,1+eps) to clip(logp-logp_k, -eps_1, +eps_2), eps1 = -log(1-eps), eps2 = log(1+eps)
        """
        # return - batch["reward"] # reparametrization trick
        #Note: detached act and logp to not re-use the graph
        logp, entropy = self.policy.logp(batch["state"],batch["act"].detach()) # get new act with requiregrad=True
        # print(T.min((logp-batch["logp"]) * (batch["rtg"] - batch["value"]), T.clamp(logp-batch["logp"],min=-np.log(1-self.eps),max=np.log(1+self.eps)) * (batch["rtg"] - batch["value"])))
    
    
        PPOLoss = T.minimum(T.exp(logp-batch["logp"].detach()) * (batch["advantage"]), T.clamp(T.exp(logp-batch["logp"].detach()),min=1-self.eps,max=1+self.eps) * (batch["advantage"]))
        # print(entropy, PPOLoss)
        # print(batch["rtg"] - batch["value"])
        #Note: minimize the negative PPO objective
        
        weight = 1.
        if "weight" in batch:
            weight = batch['weight']
        
        return weight * (- self.entrpy_scaler * entropy - PPOLoss)

    def score(self, batch:dict) -> float:
        
        if "weight" in batch:
            return T.sum(batch["reward"] * batch["weight"]) / T.sum(batch["weight"])
        return T.mean(batch["reward"]).item()
