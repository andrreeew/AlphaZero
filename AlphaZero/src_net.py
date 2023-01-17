from copy import deepcopy
import numpy as np
import game

def get_chessboard(state):
    return state[0]

def get_player(state):
    return state[-1][0][0]

def get_next_state(state, action):
    state_copy = deepcopy(state)
    chessboard, player = get_chessboard(state_copy), get_player(state_copy)
    chessboard = game.go(chessboard, action, player)

    state_copy[-1] = -state_copy[-1]
    for i in range(len(state_copy)-2, 0, -1):
        state_copy[i] = state_copy[i-1]
    state_copy[0] = chessboard

    if((not get_candidate(state_copy)) and (not check(state_copy)[0])):
        state_copy[-1] = -state_copy[-1]
        for i in range(len(state_copy)-2, 0, -1):
            state_copy[i] = state_copy[i-1]

    return state_copy


def check(state):
    return game.check(get_chessboard(state))


def get_candidate(state):
    # print(get_candidate(state), get_player(state))
    return game.get_candidate(get_chessboard(state), get_player(state))

def init_state(size=8):
    state = np.zeros((4, size, size))
    state[-1] = np.ones((size, size))
    state[0] = game.init_game(size)

    return state