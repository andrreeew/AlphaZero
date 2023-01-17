from copy import deepcopy
import numpy as np

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0

dx = [1, 1, 1, 0, 0, -1, -1, -1]
dy = [1, 0, -1, 1, -1, 1, 0, -1]

## chessboard是np.array
def get_candidate(chessboard, player):
    chessboard_size = len(chessboard[0])
    idx = np.where(chessboard==0)
    candidate_list = []
    for i in range(len(idx[0])):
        x, y = idx[0][i], idx[1][i]
        for j in range(len(dx)):
            nx, ny = x+dx[j], y+dy[j]
            is_candidate = False

            if(nx>=0 and nx<chessboard_size and ny>=0 and ny<chessboard_size and chessboard[nx][ny]==-player):
                while(True):
                    nx, ny = nx+dx[j], ny+dy[j]
                    if(not(nx>=0 and nx<chessboard_size and ny>=0 and ny<chessboard_size) or chessboard[nx][ny]==0):
                        break
                    elif(chessboard[nx][ny]==player):
                        is_candidate = True
                        break
            
            if(is_candidate):
                candidate_list.append((x, y))
                break
    return candidate_list


## chessboard走下一步，深拷贝
def go(chessboard, action, player):
    chessboard_copy = deepcopy(chessboard)
    chessboard_size = len(chessboard_copy)
    ## 执行action
    if(action is None): return chessboard_copy
    x, y = action
    chessboard_copy[x][y] = player
    for i in range(len(dx)):
        nx, ny = dx[i]+x, dy[i]+y
        if(nx>=0 and nx<chessboard_size and ny>=0 and ny<chessboard_size and chessboard_copy[nx][ny]==-player):
            flip = False
            while(True):
                nx, ny = nx+dx[i], ny+dy[i]
                if(not(nx>=0 and nx<chessboard_size and ny>=0 and ny<chessboard_size) or chessboard_copy[nx][ny]==0):
                    break
                elif(chessboard_copy[nx][ny]==player):
                    flip = True
                    break
            
            if(flip):
                while(not (nx==x and ny==y)):
                    chessboard_copy[nx][ny] = player
                    nx, ny = nx-dx[i], ny-dy[i]
    
    return chessboard_copy

#
def check(chessboard):
    ## 判断自己有无子可以下,没有则判断对方有无子可以下
    candidate = get_candidate(chessboard, 1)
    if(len(candidate)==0):
        candidate = get_candidate(chessboard, -1)


    ## 判断是否结束
    end, winner = 0, 0
    if(len(candidate)==0):
        end = 1
        white = np.sum(chessboard==COLOR_WHITE)
        black = np.sum(chessboard==COLOR_BLACK)
        # print('white', white)
        # print('black', black)
        if(white>black):
            winner = COLOR_BLACK
        elif(white<black):
            winner = COLOR_WHITE
        else:
            winner = 0

    return end, winner


def init_game(size=8):
    if(size==8):
        cheessboard = [
            [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0 , 0 , 0 , COLOR_BLACK , COLOR_WHITE , 0 , 0 , 0],
            [ 0 , 0 , 0 , COLOR_WHITE , COLOR_BLACK , 0 , 0 , 0],
            [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]]
    else:
        cheessboard = [
            [ 0 , 0 , 0 , 0],
            [ 0 , COLOR_BLACK , COLOR_WHITE , 0],
            [ 0 , COLOR_WHITE , COLOR_BLACK , 0],
            [ 0 , 0 , 0 , 0]]
    
    return np.asarray(cheessboard)
    

