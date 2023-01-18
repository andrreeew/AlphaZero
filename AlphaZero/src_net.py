from copy import deepcopy
import numpy as np
import game
cost_time = 0

def get_chessboard(state):
    return (state[0]-state[1])*get_player(state)

def get_player(state):
    return state[-1][0][0]

def get_next_state(state, action):
    state_copy = deepcopy(state)
    chessboard, player = get_chessboard(state_copy), get_player(state_copy)
    chessboard = game.go(chessboard, action, player)

    state_copy[-1] = -state_copy[-1]
    state_copy[3] = state_copy[0]
    state_copy[2] = state_copy[1]

    state_copy[0] = np.where(chessboard==get_player(state_copy), 1, 0)
    state_copy[1] = np.where(chessboard==-get_player(state_copy), 1, 0)
    

    if((not game.get_candidate(chessboard, get_player(state_copy))) and (not game.check(chessboard)[0])):
        state_copy[-1] = -state_copy[-1]
        state_copy[0], state_copy[1] = state_copy[1], state_copy[0]
        state_copy[2], state_copy[3] = state_copy[3], state_copy[2]
    
    return state_copy


def check(state):
    return game.check(get_chessboard(state))
    # return game.check(get_chessboard(state).numpy())

def get_candidate(state):
    return game.get_candidate(get_chessboard(state), get_player(state))

def init_state(size=8):
    state = np.zeros((5, size, size))
    state[-1] = np.ones((size, size))
    chessboard = game.init_game(size)
    state[0] = np.where(chessboard==1, 1, 0)
    state[1] = np.where(chessboard==-1, 1, 0)

    return state