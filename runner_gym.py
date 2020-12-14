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

from agents.fl1 import Agent

from plot import plot_scores, show_screen, moving_average

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

ENV_NAME = 'FrozenLake-v0'
env = gym.make(ENV_NAME)

class Trainer:
    def __init__(self):
        self.agent = Agent(env, optimizer_type="Adam")
        self.success = []
        self.episodes_steps_list = []
        self.episode_scores = []
    
    def train(self, episodes):
        
        episode_done = False
        reward_val = 0

        for e_i in range(episodes):
            state = env.reset()
            step_i = 0
            while step_i < 200:
                
                # perform chosen action
                action = self.agent.choose_action(state)
                # print(s, a)

                state_1, reward_val, episode_done, _ = env.step(action)
                if episode_done == True and reward_val == 0: reward_val = -1
                
                self.agent.update(action, state, state_1, reward_val, episode_done, _)
                
                # update state
                state = state_1
                step_i += 1
                if episode_done == True: break
            
            # append results onto report lists
            if episode_done == True and reward_val > 0:
                self.success.append(1)
            else:
                self.success.append(0)
            self.episodes_steps_list.append(step_i)
        
            if e_i % 100 == 0:
                success_percent = sum(self.success[-100:])
                print("success rate:{} episode:{}".format(success_percent, e_i))
                self.episode_scores.append(success_percent)
                plot_scores(self.episode_scores, ENV_NAME)
            

def main(argv): 
    print("open ai gym")
    t = Trainer()
    t.train(3000)
   
        
if __name__ == "__main__":
    main(sys.argv[1:])
