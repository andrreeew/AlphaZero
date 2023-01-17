import numpy as np
import random
import game

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
random.seed(0)

class AI(object):

    def __init__(self, chessboard_size, color, time_out=3):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
    

    def go(self, chessboard):
        actions = game.get_candidate(chessboard, self.color)
        if(not actions):
            return None
        
        scores = [np.sum(game.go(chessboard, action, self.color)==-self.color) for action in actions]
        score_max = np.max(scores)
        candidate = []
        for i in range(len(scores)):
            if(scores[i]==score_max):
                candidate.append(actions[i])
        action = random.choice(candidate)
        return action

