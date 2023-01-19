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



class AI(object):

    def __init__(self, chessboard_size, color, time_out=4, simulate_num=1000, load_state=False):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.simulate_num = simulate_num
        net = PolicyValueNet(chessboard_size)
        if(chessboard_size==6 and load_state):
            net.load_state_dict(torch.load('reversi6.params'))
        state = src_net.init_state(chessboard_size)
        root = Node(None, 1, None, None, state_now=deepcopy(state))  
        self.mct = MCT(root, net)
        
    
    def init(self):
        root = Node(state_now=deepcopy(src_net.init_state(self.chessboard_size)))  
        self.mct.root = root

    def go(self, chessboard):
        action = None

        for i in range(len(chessboard)):
            for j in range(len(chessboard)):
                if(chessboard[i][j]!=0 and src_net.get_chessboard(self.mct.get_current_state())[i][j]==0):
                    action = (i, j)
                    break
            if(action):
                break
        

        
        if(action):
            self.mct.move(action)
        
        if(not game.get_candidate(chessboard, self.color)):
            return None

        self.mct.simulate(self.simulate_num)
        action = self.mct.play()
        return action

