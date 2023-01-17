import numpy as np
import random
import time
import game
import src_net
import torch
from mcts import MCT, Node
from copy import deepcopy
from net import PolicyValueNet

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
random.seed(0)

net = None


def policy_value_fn(state):
    global net
    state_copy = torch.tensor(state).float()
    net(state_copy)
    policy = torch.ones((8, 8))
    value = torch.tensor(0)
    return policy.detach().numpy(), value.detach().numpy()


class AI(object):

    def __init__(self, chessboard_size, color, time_out=4, simulate_num=1000):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        global net
        net = PolicyValueNet(chessboard_size)

        state = src_net.init_state(chessboard_size)
        root = Node(None, 1, None, None, state_now=deepcopy(state))  
        self.mct = MCT(root, policy_value_fn, 0.1)
        
    
    def init(self):
        state = src_net.init_state(self.chessboard_size)
        root = Node(None, 1, None, None, state_now=deepcopy(state))  
        self.mct = MCT(root, policy_value_fn, 0.1)

    def go(self, chessboard):
        action = None
        for i in range(len(chessboard)):
            for j in range(len(chessboard)):
                if(chessboard[i][j]!=0 and self.mct.root.state_now[0][i][j]==0):
                    action = (i, j)
                    break
            if(action):
                break
        
        if(action):
            self.mct.move(action)
        
        if(not game.get_candidate(chessboard, self.color)):
            return None

        self.mct.simulate(100)
        action = self.mct.play()
        return action

