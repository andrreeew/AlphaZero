import numpy as np
import random
import game
from game import COLOR_BLACK, COLOR_WHITE, COLOR_NONE


class AI(object):

    def __init__(self, chessboard_size, color):
        self.chessboard_size = chessboard_size
        self.color = color
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

