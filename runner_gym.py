import os, sys
from gym.utils.seeding import np_random
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

from envs.flnew import FrozenLakeEnv
from envs.m2bpg import MoveToBeaconPygameEnv

from torch.autograd import Variable
from agents.fl1 import Agent
from plot import plot_scores, show_screen, moving_average

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

ENV_NAME = 'MoveToBeaconEnv'
# env = gym.make(ENV_NAME)

env = MoveToBeaconPygameEnv(grid_size=8)

class Trainer:
    def __init__(self):
        self.agent = Agent(env, optimizer_type="Adam")
        self.success = []
        self.episodes_steps_list = []
        self.episode_scores = []
    
    def train(self, episodes):
        
        
        

        for e_i in range(episodes):
            # print("=========episode #:{}==========".format(e_i))
            state = env.reset()
            episode_done = False
            reward_val = 0
            while not episode_done:
                
                # perform chosen action
                action = self.agent.choose_action(state)
                # print(step_i, action)

                state_1, reward_val, episode_done, p = env.step(action)

                if episode_done == True: 
                    if reward_val < 1: 
                        # print("FAIL...", reward_val)
                        self.success.append(0)
                        reward_val=-1
                    else:
                        # print("GOAL...", reward_val)
                        self.success.append(1)
                               
                self.agent.update(action, state, state_1, reward_val, episode_done, p)
                
                # update state
                state = state_1
                # time.sleep(0.1)
       
            # self.episodes_steps_list.append(step_i)
   
            if e_i % 100 == 0 and e_i != 0:
                success_percent = sum(self.success[-100:])
                print("success rate:{} episode:{}".format(success_percent, e_i))

            

def main(argv): 
    print("open ai gym")
    t = Trainer()
    t.train(30000)
   
        
if __name__ == "__main__":
    main(sys.argv[1:])
