import os, sys
import numpy as np
import time
import torch
import matplotlib
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.autograd import Variable



# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

env = gym.make('FrozenLake-v0')


class Agent(nn.Module):
    def __init__(self, observation_space_size, action_space_size):
        super(Agent, self).__init__()
        self.observation_space_size = observation_space_size
        self.hidden_size = observation_space_size
        self.l1 = nn.Linear(in_features=observation_space_size, out_features=self.hidden_size)
        self.l2 = nn.Linear(in_features=self.hidden_size, out_features=action_space_size)
        uniform_linear_layer(self.l1)
        uniform_linear_layer(self.l2)
    
    def forward(self, state):
        obs_emb = one_hot([int(state)], self.observation_space_size)
        out1 = torch.sigmoid(self.l1(obs_emb))
        return self.l2(out1).view((-1))


def one_hot(ids, nb_digits):
    """
    ids: (list, ndarray) shape:[batch_size]
    """
    if not isinstance(ids, (list, np.ndarray)):
        raise ValueError("ids must be 1-D list or array")
    batch_size = len(ids)
    ids = torch.LongTensor(ids).view(batch_size, 1)
    out_tensor = Variable(torch.FloatTensor(batch_size, nb_digits))
    out_tensor.data.zero_()
    out_tensor.data.scatter_(dim=1, index=ids, value=1.)
    return out_tensor

def uniform_linear_layer(linear_layer):
    linear_layer.weight.data.uniform_()
    linear_layer.bias.data.fill_(-0.02)


class Trainer:
    def __init__(self):
        self.agent = Agent(env.observation_space.n, env.action_space.n)
        self.optimizer = optim.Adam(params=self.agent.parameters())
        self.success = []
        self.jList = []
    
    def train(self, epoch):
        for i in range(epoch):
            s = env.reset()
            j = 0
            while j < 200:
                
                # perform chosen action
                a = self.choose_action(s)
                s1, r, d, _ = env.step(a)
                if d == True and r == 0: r = -1
                
                # calculate target and loss
                target_q = r + 0.99 * torch.max(self.agent(s1).detach()) # detach from the computing flow
                loss = F.smooth_l1_loss(self.agent(s)[a], target_q)
                
                # update model to optimize Q
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # update state
                s = s1
                j += 1
                if d == True: break
            
            # append results onto report lists
            if d == True and r > 0:
                self.success.append(1)
            else:
                self.success.append(0)
            self.jList.append(j)

            if i % 100 == 0:
                print("last 100 epoches success rate: " + str(sum(self.success[-100:])) + "%")

    def choose_action(self, s):
        if (np.random.rand(1) < 0.1): 
            return env.action_space.sample()
        else:
            agent_out = self.agent(s).detach()
            _, max_index = torch.max(agent_out, 0)
            return max_index.data.numpy().tolist()


def main(argv): 
    print("open ai gym")
    t = Trainer()
    t.train(2000)
   
        

if __name__ == "__main__":
    main(sys.argv[1:])
