import numpy as np
import random
import time
import game
import src_net
import torch
from mcts import MCT, Node
from copy import deepcopy
from net import PolicyValueNet
from game import COLOR_WHITE, COLOR_BLACK, COLOR_NONE


class AI(object):

    def __init__(self, chessboard_size, color, simulate_num=1000, net=None, c_puct=5):
        self.chessboard_size = chessboard_size
        self.color = color
        self.candidate_list = []
        self.simulate_num = simulate_num
        state = src_net.init_state(chessboard_size)
        root = Node(None, 1, None, None, state_now=deepcopy(state))  
        if(net):
            self.mct = MCT(root, policy_value_fn=net, c_puct=c_puct)
        else:
            net = PolicyValueNet(chessboard_size)
            self.mct = MCT(root, policy_value_fn=net, c_puct=c_puct)
        
    
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
        if(np.sum(chessboard==0)>=self.chessboard_size*self.chessboard_size/2.0):
            action = self.mct.play(1)
        else:
            action = self.mct.play(1e-3)
        return action

