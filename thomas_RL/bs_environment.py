import os
from typing import Tuple, Type, Optional
import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import optim
import matplotlib.pyplot as plt
import copy
from reinforcement import Reinforcement

class Environment(Reinforcement, nn.Module):
    def __init__(self, eps=0.2, lr=0.0001, dr=0.2, gamma=0.99, lam=0.95, entrpy_scaler=0.0001):
        state_dim = 4  # change this when incorporating more features
        act_dim = 1
        input_dim = state_dim
        #state_dim = 20

        self.entrpy_scaler = entrpy_scaler

        super(Environment, self).__init__(input_dim, state_dim, act_dim, eps=eps, lr=lr, dr=dr, gamma=gamma, lam=lam)

        # Black-Scholes parameters
        self.r = 0.007  # risk-free rate
        self.sigma = 0.2  # constant volatility

        self.Tunit = 1 / 252  # time unit (1 day)
        self.dt = self.Tunit / 10  # time step

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def reset(self):
        self.iat = 0  # initial time step
        self.S0 = 450  # initial stock price
        self.T = 20  # terminal time
        self.K = 450  # strike price

        # Compute initial option price using Black-Scholes formula
        self.option = self.black_scholes_put(self.S0, self.K, self.T * self.Tunit, self.r, self.sigma)
        self.cash = self.option  # initial cash position
        self.delta = 0  # initial delta (hedge ratio)

        self.w0 = self.cash  # initial wealth

        self.moneyness = self.S0 / self.K - 1  # moneyness

        # Initial state: [moneyness, sqrt(time to maturity), delta, dd]
        self.state = T.tensor([self.moneyness, np.sqrt(self.T * self.Tunit), self.delta, -0.5]).to(self.device)
        return

    def evolve(self, action) -> Tuple[T.Tensor, T.Tensor, bool]:
        oldstate = self.state.clone().detach()

        # Update cash position
        self.cash *= np.exp(self.r * self.dt)
        #print(action)
        self.cash -= self.S0 * (action[0].item() + np.abs(action[0].item() * 0.01))  # transaction cost
        self.delta += action[0].item()  # update delta
        self.iat += 1  # increment time step


        temp = [self.S0, self.option]
        # Simulate stock price using Black-Scholes dynamics
        Z = np.random.normal(0, 1)  # standard normal random variable
        self.S0 *= np.exp((self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)

        # Compute new option price
        self.option = self.black_scholes_put(self.S0, self.K, (self.T - self.iat) * self.Tunit, self.r, self.sigma)

        temp[0] = self.S0 - temp[0]
        temp[1] = self.option - temp[1]
        dd = temp[1] / temp[0]
        
        
        # Compute reward (hedging error)
        reward = T.minimum(
            (T.tensor(self.cash + self.S0 * self.delta - self.option / self.w0) - 1.).to(self.device),
            T.zeros(1, dtype=T.double).to(self.device)
        )

        # Check if episode is done
        done = False
        if (self.iat == self.T) or (reward.item() < -2):
            done = True
        else:
            # Update state
            self.state = T.tensor([self.moneyness, np.sqrt((self.T - self.iat) * self.Tunit), self.delta, dd]).to(self.device)


        return oldstate, reward, done

    def black_scholes_put(self, S, K, ttm, r, sigma):
        """
        Black-Scholes formula for a European put option.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * ttm) / (sigma * np.sqrt(ttm))
        d2 = d1 - sigma * np.sqrt(ttm)
        d1 = T.tensor(d1)
        d2 = T.tensor(d2)
        put_price = K * np.exp(-r * ttm) * D.Normal(0, 1).cdf(-d2) - S * D.Normal(0, 1).cdf(-d1)
        return put_price

    def objective(self, batch: dict) -> T.tensor:
        """
        PPO objective function.
        """
        logp, entropy = self.policy.logp(batch["state"], batch["act"].detach())
        PPOLoss = T.minimum(
            T.exp(logp - batch["logp"].detach()) * batch["advantage"],
            T.clamp(T.exp(logp - batch["logp"].detach()), min=1 - self.eps, max=1 + self.eps) * batch["advantage"]
        )
        weight = batch.get("weight", 1.0)
        return weight * (-self.entrpy_scaler * entropy - PPOLoss)

    def score(self, batch: dict) -> float:
        """
        Compute the average reward.
        """
        if "weight" in batch:
            return T.sum(batch["reward"] * batch["weight"]) / T.sum(batch["weight"])
        return T.mean(batch["reward"]).item()