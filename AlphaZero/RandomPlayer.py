import numpy as np
import random
import time
import game
import src_net
from mcts import MCT, Node
from copy import deepcopy

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
random.seed(0)

def policy_value_fn(state):
    policy = np.ones((8, 8))
    value = 0
    return policy, value 


class AI(object):

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
    

    def go(self, chessboard):
        actions = game.get_candidate(chessboard, self.color)
        if(not actions):
            return None
        return random.choice(actions)


