from game import get_candidate, check, go, init_game, COLOR_BLACK, COLOR_WHITE
import numpy as np
from tqdm import tqdm
from Agent.MCTSPlayer import AI as player_mcts
from Agent.GreedyPlayer import AI as player_greedy
from Agent.RandomPlayer import AI as player_random
from net import PolicyValueNet
import torch


def compete(player_white, player_black, size=8):
    chessboard = init_game(size)
    player_now = COLOR_WHITE
    while(True):
        end, winner = check(chessboard)
        if(end):
            # print(chessboard)
            # print('---------------------------')
            return winner
        
        if(player_now==COLOR_BLACK):
            action = player_black.go(chessboard)
        else:
            action = player_white.go(chessboard)
        
        chessboard = go(chessboard, action, player_now)
        # print(chessboard)
        # print('black:', np.sum(chessboard==COLOR_BLACK))
        # print('white:', np.sum(chessboard==COLOR_WHITE))
        # print('-----------------------------')
        player_now = COLOR_BLACK if player_now==COLOR_WHITE else COLOR_WHITE



def compete_net(net1, net2, size=8, num_game=20, simulate_num=300):
    cnt1, cnt2 = 0, 0
    # games = tqdm(range(num_game))
    for i in range(num_game):
    # for i in enumerate(games):
        if(i%2==0):
            player_white = player_mcts(size, COLOR_WHITE, simulate_num=simulate_num, net=net1)
            player_black = player_mcts(size, COLOR_BLACK, simulate_num=simulate_num, net=net2)
        else:
            player_black = player_mcts(size, COLOR_BLACK, simulate_num=simulate_num, net=net1)
            player_white = player_mcts(size, COLOR_WHITE, simulate_num=simulate_num, net=net2)
        
        winner = compete(player_white, player_black, size)
        if(i%2==0):
            if(winner==COLOR_WHITE):
                cnt1 += 1
            elif(winner==COLOR_BLACK):
                cnt2 += 1
        else:
            if(winner==COLOR_WHITE):
                cnt2 += 1
            elif(winner==COLOR_BLACK):
                cnt1 += 1
        # games.set_postfix(win=cnt1, lose=cnt2)
    
    return cnt1, cnt2


if __name__ == '__main__':
    SIZE = 6
    
    net1 = PolicyValueNet(SIZE)
    net1.load_state_dict(torch.load('reversi'+str(SIZE)+'.params'))
    
    # net2 = PolicyValueNet(SIZE)
    # net2.load_state_dict(torch.load('reversi'+str(SIZE)+'_tmp.params'))
    
    
    player_black = player_mcts(SIZE, COLOR_BLACK, simulate_num=500, net=net1)
    # player_black = player_greedy(SIZE, COLOR_BLACK)
    # player_black = player_random(SIZE, COLOR_BLACK)

    # player_white = player_mcts(SIZE, COLOR_WHITE, simulate_num=500, net=net1)
    # player_white = player_greedy(SIZE, COLOR_WHITE)
    player_white = player_random(SIZE, COLOR_WHITE)
    

    games = tqdm(range(20))

    white_cnt, black_cnt = 0, 0
    for game in enumerate(games):
        if(isinstance(player_white, player_mcts)):
            player_white.init()
        if(isinstance(player_black, player_mcts)):
            player_black.init()

        winner = compete(player_white, player_black, SIZE)
        if(winner==COLOR_BLACK):
            black_cnt += 1
            # print('Black win')
        else:
            white_cnt += 1
            # print('White win')
        games.set_postfix(white=white_cnt, black=black_cnt)

    print('Black win', black_cnt)
    print('White win', white_cnt)




