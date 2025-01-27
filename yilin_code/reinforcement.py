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
from policy_network import PolicyNet
from baseline_network import BaselineNet
import copy

class Reinforcement(nn.Module):
    def __init__(self, state_dim, act_dim, eps=0.2, lr=0.0001, dr=0.2, gamma=0.99, lam=0.95, bufsize=4096, rollsize=256, model_dir="./model"):
        super(Reinforcement, self).__init__()

        self.eps = eps
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.gamma = gamma
        self.lam = lam
        self.pho = np.exp(np.log(0.1) / (bufsize/rollsize-1))

        self.state = None #T.tensor.new_empty(state_dim)

        self.replaybuffer = {}
        self.buffersize = bufsize
        self.rollsize = rollsize

        self.policy = PolicyNet(state_dim, act_dim, lr, dr, model_dir=model_dir, name="Policy").double() #change precision here
        self.baseline = BaselineNet(state_dim, lr, dr, model_dir=model_dir, name="Baseline").double()

        self.mse = nn.MSELoss() # initialize once to train baseline

        self.epochs = 0
        self.best = -np.inf
        self.trainloss = []
        self.trainscore = []
        self.validloss = [] #per epoch
        self.validscore = []

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    #TODO: remember to deal with self.state=None (note in doc)
    def reset(self):
        raise NotImplementedError

    def evolve(self, action) -> Tuple[T.Tensor, T.Tensor, bool]: #virtual: outputs old_state, reward/loss, done ; update current state internally
        raise NotImplementedError

    def simulate(self, max_len=np.inf, max_iter=np.inf, eval=False):
        traj = {}
        traj['state'] = []
        traj['act'] = []
        traj['logp'] = []
        # traj['entropy'] = []
        traj['reward'] = []
        traj['rtg'] = [] #reward to go
        

        done = False
        length = 0
        self.reset()

        while (not done) and (length < max_iter): #simulate one step forward with `evolve`
            traj['state'] += [self.state]
            if not eval:
                act, logp = self.policy.sample(self.state)
            else:
                act, logp = self.policy.evaluate(self.state) #TODO: change logp to some kind of interval accordingly

            # act = full_range(act)

            traj['act'] += [act]
            traj['logp'] += [logp]
            _, reward, done = self.evolve(act)
            # print(done)
            traj['reward'] += [reward]
            length += 1

        traj['rtg'] = [0 for i in range(length)]
        for i in reversed(range(length)):
            traj['rtg'][i] = traj['reward'][i].detach() + self.gamma*(0 if (i+1 >= length) else traj['rtg'][i+1])
            
        self.baseline.eval()
        val = self.baseline(T.stack(traj["state"])).detach()
        traj['value'] = val
        
        
        # GAE advantage
        
        delta = [0 for i in range(length)]
        for i in reversed(range(length)):
            delta[i] = traj['reward'][i].detach() - val[i] + self.gamma*(0 if (i+1 >= length) else val[i+1])
        
        f = self.lam * self.gamma
        traj['advantage'] = [0 for i in range(length)]
        for i in reversed(range(length)):
            traj['advantage'][i] = delta[i] + f*(0 if (i+1 >= length) else traj['advantage'][i+1])
        
        traj['weight'] = T.ones_like(val)

        for k in ('state', 'act', 'logp', 'reward', 'rtg', 'value', 'advantage', 'weight'):
            traj[k] = traj[k][:min(length,max_len)]

        return traj, min(length,max_len)

    def objective(self, batch: dict) -> T.tensor:
        raise NotImplementedError

    def optimize(self, batch_size, iterations):
        #moved from initialization to here since cannot share through batchs (need to be initialized empty each epoch)
        batch = {}
        batch['state'] = []
        batch['act'] = [] #Note: we dont need to store action here if logprob is stored, however, lets keep it here for future changes.(state is necessary for baseline training
        batch['logp'] = []
        batch['reward'] = []
        batch['rtg'] = [] #reward to go
        batch['value'] = []
        batch['advantage'] = []
        batch['weight'] = []
        #DONE: How to implement reparameterized sample action in a batch without breaking the gradient graph?
        #Solution foureplaybufferuse expanding list to store variable with gradient (do not preallocate). List Append definitely works; for our convenience, lets try += here and see if it works

        counter = 0

        # fetch the batch through simulation
        while counter < batch_size:
            #print(batch_size)
            #print(counter)
            t, l = self.simulate(max_len= batch_size - counter) #set eval=True for debug the network (using mean only)
            for k in ('state', 'act', 'logp', 'reward', 'rtg', 'value', 'advantage', 'weight'):
                batch[k] += t[k]
            counter += l
            
        if 'weight' in self.replaybuffer:
            self.replaybuffer['weight'] *= self.pho

        for k in ('state', 'act', 'logp', 'reward', 'rtg', 'value', 'advantage', 'weight'):
            
            batch[k] = T.stack(batch[k])
            
            if k not in self.replaybuffer:
                self.replaybuffer[k] = batch[k]
            else:
                self.replaybuffer[k] = T.cat([self.replaybuffer[k],batch[k]])
            self.replaybuffer[k] = self.replaybuffer[k][-self.buffersize:]
        
        

        # Already detached!: batch['rtg'] = batch['rtg'].detach() #otherwise will re-use model graph
        # print(batch["state"])

#         merge these two training step into one
#         #optimize baseline net
#         self.baseline.train()
#         for i in range(iterations):
#             rw = self.baseline(batch["state"]) #DEBUG: batch input
#             loss = self.mse(rw, batch["rtg"])
#             self.baseline.optimizer.zero_grad()
#             loss.backward()
#             self.baseline.optimizer.step()

#         self.baseline.eval()

#         #DONE: add baseline calculation here (detached)
#         val = self.baseline(batch["state"])
#         batch['value'] = val.detach()

#         # optimize policy net; DONE: Consider to add multiple step descend for each iteration
#         for i in range(iterations): #DEBUG: Something is re-using!
#             # print(i)
#             loss = self.objective(batch)
#             self.policy.optimizer.zero_grad()
#             loss.mean().backward()
#             self.policy.optimizer.step()

        for i in range(iterations):
            # print(i)
            self.baseline.train()
            rw = self.baseline(self.replaybuffer["state"]) #DEBUG: batch input
            loss = self.mse(rw, self.replaybuffer["rtg"])
            self.baseline.optimizer.zero_grad()
            loss.backward()
            self.baseline.optimizer.step()

            # self.baseline.eval()
            # val = self.baseline(batch["state"])
            # batch['value'] = val.detach()

            loss = self.objective(self.replaybuffer)
            self.policy.optimizer.zero_grad()
            loss.mean().backward()
            self.policy.optimizer.step()


        # loss = self.objective(batch)
        # self.trainloss += [loss.sum().item()]
        # self.trainscore += [self.score(batch)]

        return self

    def score(self, batch:dict) -> float:
        raise NotImplementedError
        
        
    def get_eval_batch(self, batch_size=512):
        
        # TODO: should make this abstract function in the future
        batch = {}
        batch['state'] = []
        batch['act'] = [] #Note: we dont need to store action here if logprob is stored, however, lets keep it here for future changes.(state is necessary for baseline training
        batch['logp'] = []
        batch['reward'] = []
        batch['rtg'] = [] #reward to go
        batch['value'] = []
        batch['advantage'] = []
        #DONE: How to implement reparameterized sample action in a batch without breaking the gradient graph?
        #Solution foureplaybufferuse expanding list to store variable with gradient (do not preallocate). List Append definitely works; for our convenience, lets try += here and see if it works
        
        counter = 0

        # fetch the batch through simulation
        while counter < batch_size:
            t, l = self.simulate(max_len=batch_size-counter) #set eval=True for debug the network (using mean only)
            for k in ('state', 'act', 'logp', 'reward', 'rtg', 'value', 'advantage'):
                batch[k] += t[k]
            counter += l

        for k in ('state', 'act', 'logp', 'reward', 'rtg', 'value', 'advantage'):
            batch[k] = T.stack(batch[k])
        
        return batch

    def evaluate(self, batch_size=512):
        
        eval_batch = self.get_eval_batch(batch_size)
        # compute loss and score
        loss = self.objective(eval_batch)
        score = self.score(eval_batch)

        return loss.mean().item(), score

    def train(self, epochs=100, train_iterations=8, val_batchs=512, save_epochs=10, verbose=2):
        
        for i in range(epochs):
            self.optimize(self.rollsize, train_iterations)
            loss, score = self.evaluate(val_batchs)
            # print((score, self.best))
            if (score > self.best) and (self.epochs > 1):
                self.best = score
                self.export(suffix="_best")

            if (self.epochs % save_epochs == 0):
                self.export(suffix=f"_save_{self.epochs}")
                
            tloss = self.objective(self.replaybuffer).mean().item()
            tscore = self.score(self.replaybuffer)

            if verbose == 2:
                print("Epoch",i+1,":",f"valid - loss: {loss}, score: {score}; insample - loss: {tloss}, score: {tscore}")
            elif verbose == 1:
                print("Epoch",i+1,":",f"valid - loss: {loss}, score: {score}; insample - loss: {tloss}, score: {tscore}", end='\r')

            self.validloss += [loss]
            self.validscore += [score]
            self.trainloss += [loss]
            self.trainscore += [score]

            self.epochs += 1

        return self


    def get_insample_loss(self):
        return copy.deepcopy(self.trainloss)

    def get_insample_score(self):
        return copy.deepcopy(self.trainscore)

    def get_validation_loss(self):
        return copy.deepcopy(self.validloss)

    def get_validation_score(self):
        return copy.deepcopy(self.validscore)

    def export(self, suffix : str = None):
        if suffix is None:
            return copy.deepcopy(self.policy), copy.deepcopy(self.baseline)
        else:
            #print(suffix)
            self.policy.save(suffix=suffix)
            self.baseline.save(suffix=suffix)
        return self.policy, self.baseline

    def load(self, suffix : str = None, suffix_b : Optional[str] = None, policy : Optional[Type[PolicyNet]] = None, baseline : Optional[Type[BaselineNet]] = None):
        op, ob = copy.deepcopy(self.policy), copy.deepcopy(self.baseline)

        if suffix is None:
            self.policy, self.baseline = policy, baseline
        else:
            self.policy.load(suffix=suffix)
            self.baseline.load(suffix=suffix)
        return op, ob