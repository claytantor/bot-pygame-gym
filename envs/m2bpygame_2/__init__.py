import sys
from contextlib import closing

import numpy as np
import pygame

from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

from games.m2b import MoveToBeacon
from pygame.constants import KEYDOWN, KEYUP, K_F15

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS_A = {
    "4_4": [
        [1,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,2]
    ],
}

class MoveToBeaconEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, map_name="4_4"):
    
        desc = MAPS_A[map_name]
        self.desc = desc = np.asarray(desc, dtype='int16')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        self.screen_dim = (self.ncol*32, self.nrow*32)
        
        # self.screen = pygame.display.set_mode(self.screen_dim, 0, 32)
        # self.clock = pygame.time.Clock()

        self.game = MoveToBeacon(self.desc)
        # self.game.init()

        # pygame.display.set_mode((1, 1), pygame.NOFRAME)
        # self.game._setup()
        # self.game.init() #this is the games setup/init

        nA = 4
        nS = nrow * ncol
        self.reward_range = (0, 1)
        isd = np.array(desc == 1).astype('float64').ravel()
        isd /= isd.sum()
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newval = desc[newrow, newcol]

            done = newval in [2]
            reward = float(newval in [2])
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    map_val = desc[row, col]

                    if map_val in [2]:
                        li.append((1.0, s, 0, True))
                    else:
                        li.append((1., *update_probability_matrix(row, col, a)))

        super(MoveToBeaconEnv, self).__init__(nS, nA, P, isd)

    def get_map(self):
        return self.desc

    # def pygame_setup(self):
    #     """
    #     Setups up the pygame env, the display and game clock.
    #     """
    #     pygame.init()
    #     self.screen = pygame.display.set_mode(self.screen_dim, 0, 32)
    #     self.clock = pygame.time.Clock()

    def pygame_set_action(self, action):
        """
        Pushes the action to the pygame event queue.
        """
        if action is None:
            self.action = self.NOOP


        kd = pygame.event.Event(KEYDOWN, {"key": action})
        # ku = pygame.event.Event(KEYUP, {"key": last_action})

        pygame.event.post(kd)

    def pygame_draw_frame(self, draw_screen=True):
        """
        Decides if the screen will be drawn too
        """

        if draw_screen == True:
            pygame.display.update()


    def render(self, mode='human'):
        if mode in ['human','ansi']:
            outfile = StringIO() if mode == 'ansi' else sys.stdout

            row, col = self.s // self.ncol, self.s % self.ncol
            desc = self.desc.tolist()
            desc = [[c.decode('utf-8') for c in line] for line in desc]
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
            if self.lastaction is not None:
                outfile.write("  ({})\n".format(
                    ["Left", "Down", "Right", "Up"][self.lastaction]))
            else:
                outfile.write("\n")
            outfile.write("\n".join(''.join(line) for line in desc)+"\n")

            if mode != 'human':
                with closing(outfile):
                    return outfile.getvalue()
        elif mode in ['rgb']:
            row, col = self.s // self.ncol, self.s % self.ncol 
            desc = self.desc.tolist()
            if self.lastaction is not None:
                self.pygame_set_action(self.lastaction)
                self.game.step(30)
                self.pygame_draw_frame()
                # self.game.step(30)    

