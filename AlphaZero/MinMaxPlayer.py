from itertools import count
import numpy as np
import random
import time
from copy import deepcopy

dx = [1, 1, 1, 0, 0, -1, -1, -1]
dy = [1, 0, -1, 1, -1, 1, 0, -1]
class Game:
    ## chessboard鏄痭p.array
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

    # 鑾峰彇妫嬬洏涓妏layer鐨勪綅缃紝杩斿洖鐨勬槸0-1鐨勭煩闃碉紝1琛╬layer鏈変笅鍦ㄩ偅锛�0琛ㄧず娌℃湁
    def get_position(chessboard, player):
        return np.where(chessboard==player, 1, 0)

    ## chessboard璧颁笅涓€姝ワ紝娣辨嫹璐�
    def go(chessboard, action, player):
        chessboard_copy = deepcopy(chessboard)
        chessboard_size = len(chessboard_copy)
        ## 鎵цaction
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
    

    def check(chessboard):
        ## 鍒ゆ柇鑷繁鏈夋棤瀛愬彲浠ヤ笅,娌℃湁鍒欏垽鏂鏂规湁鏃犲瓙鍙互涓�
        candidate = Game.get_candidate(chessboard, 1)
        if(len(candidate)==0):
            candidate = Game.get_candidate(chessboard, -1)
    
    
        ## 鍒ゆ柇鏄惁缁撴潫
        end, winner = 0, 0
        if(len(candidate)==0):
            end = 1
            white = np.sum(chessboard==1)
            black = np.sum(chessboard==-1)
            # print('white', white)
            # print('black', black)
            if(white>black):
                winner = -1
            elif(white<black):
                winner = 1
            else:
                winner = 0

        return end, winner

class Strategy:  
    ## 绛栫暐涓€锛歮in鈥攎ax, 鍙傛暟锛歟lavation_fn璇勪及鍑芥暟锛宒epth鎼滅储鐨勬繁搴�
    def min_max_search(chessboard, color, elavation_fn=None, depth=4):
        player = color
        infinity = 1e8

        def value(winner, player):
            if(winner==0):
                return 1e6
            elif(winner==player):
                return 1e7
            return -1e7

        # 褰撳墠player涓�
        def max_value(chessboard, alpha, beta, depth):
            end, winner = Game.check(chessboard)
            if end:
                return value(winner, player), None
            
            ## 濡傛灉娣卞害涓�0锛屼緷鎹瘎浼板嚱鏁拌繑鍥炲綋鍓嶆鐩樹腑player鐨勮瘎浼�
            if(depth==0):
                return elavation_fn.elevate(chessboard, player), None

            v, move = -infinity, None
            candidate = Game.get_candidate(chessboard, player)
            # 濡傛灉娌℃湁妫嬪彲浠ヨ蛋锛屽垯绌鸿蛋
            if(len(candidate)==0):
                return min_value(Game.go(chessboard, None, player), alpha, beta, depth-1)    
            # 濡傛灉鏈夋瀛愬垯閬嶅巻
            else:
                for a in candidate:
                    v2, _ = min_value(Game.go(chessboard, a, player), alpha, beta, depth-1)
                    if(v2>v):
                        v, move = v2, a
                    if(v>=beta):
                        return v, move
                    alpha = max(alpha, v)

            return v, move

        # 瀵规墜-player涓�
        def min_value(chessboard, alpha, beta, depth):
            end, winner = Game.check(chessboard)
            if end:
                return value(winner, player), None
            
            ## 濡傛灉娣卞害涓�0锛屼緷鎹瘎浼板嚱鏁拌繑鍥炲綋鍓嶆鐩樹腑player鐨勮瘎浼�
            if(depth==0):
                return elavation_fn.elevate(chessboard, player), None

            v, move = infinity, None
            candidate = Game.get_candidate(chessboard, -player)
            # 濡傛灉娌℃湁妫嬪彲浠ヨ蛋锛屽垯绌鸿蛋
            if(len(candidate)==0):
                return max_value(Game.go(chessboard, None, player), alpha, beta, depth-1)
            # 濡傛灉鏈夋瀛愬垯閬嶅巻
            else:
                for a in candidate:
                    v2, _ = max_value(Game.go(chessboard, a, -player), alpha, beta, depth-1)
                    if(v2<v):
                        v, move = v2, a
                    if(v<=alpha):
                        return v, move
                    beta = min(beta, v)    

            return v, move

        return max_value(chessboard, -infinity, +infinity, depth)[1]


    ## 绛栫暐浜岋細璐績
    def greedy(chessboard, color):
        return 0


    ## 绛栫暐涓夛細闅忔満
    def random(chessboard, color):
        candidate_list = Game.get_candidate(chessboard, color)
        action = random.choice(candidate_list)
        if(action==(0, 0) or action==(7, 0) or action==(0, 7) or action==(7, 7)):
            action = random.choice(candidate_list)
        if(action==(0, 0) or action==(7, 0) or action==(0, 7) or action==(7, 7)):
            action = random.choice(candidate_list)
        if(action==(0, 0) or action==(7, 0) or action==(0, 7) or action==(7, 7)):
            action = random.choice(candidate_list)
        
        return action


    ## 绛栫暐鍥涳細鏉冮噸
    def weight(chessboard, color):
        candidate_list = Game.get_candidate(chessboard, color)
        for v in candidate_list:
            if(v in ((0, 1), (1,0), (0, 6), (1, 7), (6, 0), (7, 1), (7, 6), (6, 7))):
                return v

        action = random.choice(candidate_list)
        if(action==(0, 0) or action==(7, 0) or action==(0, 7) or action==(7, 7)):
            action = random.choice(candidate_list)
        if(action==(0, 0) or action==(7, 0) or action==(0, 7) or action==(7, 7)):
            action = random.choice(candidate_list)
        if(action==(0, 0) or action==(7, 0) or action==(0, 7) or action==(7, 7)):
            action = random.choice(candidate_list)
        
        return action


class elevation_fn:

    def __init__(self, params=None):
        self.frontier_factor = params['frontier']
        self.stability_factor = params['stability']
        self.mobility_factor = params['mobility']
        self.position_factor = params['position']
        self.count_factor = params['count']

        # v = [-50, 50, -10, 1, 20, 1, 20,  4,  4,  4]
        v = [-4000, 40, -2, 0.1, 0.1, 0.1, 0.1,  0,  0,  0]
        self.position_weight = np.asarray([
            [v[0], v[1], v[3], v[5], v[5], v[3], v[1], v[0]],
            [v[1], v[2], v[4], v[6], v[6], v[4], v[2], v[1]],
            [v[3], v[4], v[7], v[8], v[8], v[7], v[4], v[3]],
            [v[5], v[6], v[8], v[9], v[9], v[8], v[6], v[5]],
            [v[5], v[6], v[8], v[9], v[9], v[8], v[6], v[5]],
            [v[3], v[4], v[7], v[8], v[8], v[7], v[4], v[3]],
            [v[1], v[2], v[4], v[6], v[6], v[4], v[2], v[1]],
            [v[0], v[1], v[3], v[5], v[5], v[3], v[1], v[0]]
            ])
        return None
    
    
    def adjust_position_weight(self, chessboard):
        if(chessboard[0][1]!=0 and chessboard[1][0]!=0 and chessboard[0][0]==0):
            self.position_weight[1][1] = 60
        if(chessboard[0][6]!=0 and chessboard[1][7]!=0 and chessboard[0][7]==0):
            self.position_weight[1][6] = 60
        if(chessboard[6][0]!=0 and chessboard[7][1]!=0 and chessboard[7][0]==0):
            self.position_weight[6][1] = 60
        if(chessboard[6][7]!=0 and chessboard[7][6]!=0 and chessboard[7][7]==0):
            self.position_weight[6][6] = 60
    

    
    def elevate(self, chessboard, color):
        frontier_cnt = {-1:0, 1:0} if self.frontier_factor==0 else self.frontier_cnt(chessboard)
        # 鍓嶆部瀛愬緱鍒�
        frontier_score = 1.0*(frontier_cnt[-color]-frontier_cnt[color])
        
        boundary_cnt = {-1:0, 1:0} if self.stability_factor==0 else self.boundary_cnt(chessboard)
        #绋冲畾瀛愬緱鍒�
        stability_score = 1.0*(boundary_cnt[-color]-boundary_cnt[color])

        #琛屽姩鍔涘緱鍒�
        mobility_score = 0 if self.mobility_factor==0 else len(Game.get_candidate(chessboard, color))-len(Game.get_candidate(chessboard, -color))


        #浣嶇疆鏉冮噸寰楀垎 np.multiply鎸夌収浣嶇疆鐩镐箻锛岃€岄潪鐭╅樀涔樻硶
        if(self.position_factor!=0):
            self.adjust_position_weight(chessboard)
        position_score = 0 if self.position_factor==0 else np.sum(np.multiply(chessboard, self.position_weight))*color


        ##鏁扮洰寰楀垎
        count_score = np.sum(chessboard)*(-color)


        # return frontier_score+stability_score+mobility_score
        return self.stability_factor*stability_score+self.frontier_factor*frontier_score+self.mobility_factor*mobility_score+self.position_factor*position_score+self.count_factor*count_score
    

    ## 杈圭晫涓婄殑绋冲畾瀛愭暟鐩紝杩斿洖{1:cnt(1), -1:cnt(-1)}
    def boundary_cnt(self, chessboard):
        chessboard_size = len(chessboard)
        ## used鐭╅樀璁板綍鏈夋病鏈夌ǔ瀹氬瓙鍦ㄨ浣嶇疆涓�
        used = {1:np.zeros([chessboard_size, chessboard_size]), -1:np.zeros([chessboard_size, chessboard_size])}
        ## 鏋氫妇杈圭晫
        for (sx, sy) in [(0, 0), (chessboard_size-1, 0), (0, chessboard_size-1), (chessboard_size-1, chessboard_size-1)]:
            if(chessboard[sx][sy]==0):
                continue
            color = chessboard[sx][sy]
            x_step = 1 if sx==0 else -1
            y_step = 1 if sy==0 else -1

            ## 鏈濅袱涓柟鍚戣鏁�
            x, y = sx, sy
            while(x>=0 and x<chessboard_size):
                if(chessboard[x][y]==color):
                    used[color][x][y] = 1
                    x += x_step
                else: 
                    break

            x, y = sx, sy
            while(y>=0 and y<chessboard_size):
                if(chessboard[x][y]==color):
                    used[color][x][y] = 1
                    y += y_step
                else:
                    break
            
        return {1:np.sum(used[1]), -1:np.sum(used[-1])}


    ## 鍓嶆部瀛�(鍗虫病鏈夎鍖呰９鐨勫瓙)鏁扮洰锛岃繑鍥瀧1:cnt(1), -1:cnt(-1)}
    def frontier_cnt(self, chessboard):
        chessboard_size = len(chessboard)
        ## used鐭╅樀璁板綍鏈夋病鏈夊墠娌垮瓙鍦ㄨ浣嶇疆涓�
        used = {1:np.zeros([chessboard_size, chessboard_size]), -1:np.zeros([chessboard_size, chessboard_size])}

        for x in range(chessboard_size):
            for y in range(chessboard_size):
                if(chessboard[x][y]!=0):
                    continue
                ## 鑻ヨ鐐逛负绌猴紝鏋氫妇鍏跺懆鍥寸殑鍓嶆部瀛�
                for dx, dy in [(0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0)]:
                    nx, ny = x+dx, y+dy
                    if(nx>=0 and nx<chessboard_size and ny>=0 and ny<chessboard_size and chessboard[nx][ny]!=0):
                        used[chessboard[nx][ny]][nx][ny] = 1

        return {1:np.sum(used[1]), -1:np.sum(used[-1])}


COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
random.seed(0)
#don't change the class name

class AI(object):

    #chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        #You are white or black
        self.color = color
        #the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []


    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        self.candidate_list = Game.get_candidate(chessboard, self.color)
        if(len(self.candidate_list)==0):
            return None

        # random.shuffle(self.candidate_list)
        for v in self.candidate_list:
            if(v not in ((0, 0), (7, 0), (0, 7), (7, 7))):
                self.candidate_list.append(v)
                break   
               
        action = random.choice(self.candidate_list)
        if(not (action==(0, 0) or action==(7, 0) or action==(0, 7) or action==(7, 7))):
            self.candidate_list.append(action)


        n_step = np.sum(chessboard==0)
        ## 鍓嶉潰鏄箣鍓嶇殑鍙傛暟锛屽钩鍙颁笂閭ｇ増鐨�
        #params1 = {"frontier": 11.954225960234385, "stability": 8.098003662195548, "mobility": 3.0424910065673663, "position": 1.2423956868195003, "count":0}
        #params2 = {"frontier": 23.17214825542879, "stability": 23.73189914011259, "mobility": 25.99035728330399, "position": 0.750315857841584, "count": 13}
       ## params1, params2 = [{"frontier": 22.106085977374438, "stability": 19.3407607104084, "mobility": 28.86325343766262, "position": 2, "count": 9.068628026464808}, {"frontier": 8.459923771431066, "stability": 25.008380290593895, "mobility": 15.956910681654819, "position": 1.1, "count": 18.830096880391448}]
        # params1, params2 = [{"frontier": 22.106085977374438, "stability": 22.3407607104084, "mobility": 28.86325343766262, "position": 2, "count": 5.068628026464808}, {"frontier": 10.459923771431066, "stability": 25.008380290593895, "mobility": 15.956910681654819, "position": 1.1, "count": 18.830096880391448}]
        # params1, params2 = [{"frontier": 24.719574257439444, "stability": 20.672438796712367, "mobility": 21.742443381332233, "position": 2.0, "count": 6.187883214483084}, {"frontier": 1.1525031759577198, "stability": 27.999693950170666, "mobility": 25.325101900472205, "position": 1.0, "count": 11.313738183101771}]
        params1, params2 = [{"frontier": 0, "stability": 0, "mobility": 0, "position": 0, "count": 0}, {"frontier": 0, "stability": 0, "mobility": 0, "position": 0, "count": 0}]

        if(n_step>=40):
            # action = Strategy.random(chessboard, self.color)
            action = Strategy.weight(chessboard, self.color)
        elif(n_step>=16):
            if(len(self.candidate_list)>=17):
                action = Strategy.min_max_search(chessboard, self.color, elevation_fn(params1), 3)
                self.candidate_list.append(action)
            
            action = Strategy.min_max_search(chessboard, self.color, elevation_fn(params1), 4)
        else:
            action = Strategy.min_max_search(chessboard, self.color, elevation_fn(params2), 20)
   
        self.candidate_list.append(action)
        return self.candidate_list[-1]