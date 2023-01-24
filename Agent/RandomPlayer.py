import numpy as np
import random
import game

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
random.seed(0)

class AI(object):

    def __init__(self, chessboard_size, color):
        self.chessboard_size = chessboard_size
        self.color = color
        self.candidate_list = []
    

    def go(self, chessboard):
        actions = game.get_candidate(chessboard, self.color)
        if(not actions):
            return None
        return random.choice(actions)


