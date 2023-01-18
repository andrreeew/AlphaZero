import random
import time
from typing import Tuple, List, Set, Optional
# from functools import lru_cache
import numpy as np


COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0

# random.seed(0)

def relu(x):
    x[x < 0] = 0
    return x


def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


class Conv2d:

    def __init__(self, weight, bias, stride: int, padding: int) -> None:
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.pad = padding

    def forward(self, x):
        ax, ca, ih, iw = x.shape
        ay, _, oh, ow = self.weight.shape
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')
        ih, iw = ih + 2 * self.pad, iw + 2 * self.pad
        out_h, out_w = int((ih - oh) / self.stride + 1), int((iw - ow) / self.stride + 1)
        shape, strides = (ca, oh, ow, ax, out_h, out_w), (ih * iw, iw, 1, ca * ih * iw, self.stride * iw, self.stride)
        strides = x.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides).reshape(
            (ca * oh * ow, ax * out_h * out_w))
        res = self.weight.reshape(ay, -1).dot(x_stride) + self.bias.reshape(-1, 1)
        res.shape = (ay, ax, out_h, out_w)
        return np.ascontiguousarray(res.transpose(1, 0, 2, 3))


class BatchNorm:
    def __init__(self, weight, bias, mean, var) -> None:
        self.weight = weight
        self.bias = bias
        self.mean = mean
        self.var = var

    def forward(self, x):
        # print("bat", self.mean[None, :, None, None].shape)
        tmp = (x - self.mean[None, :, None, None]) / np.sqrt(self.var[None, :, None, None])
        # x1 = self.weight[None, :, None, None] * tmp
        # x2 = self.bias[None, :, None, None]
        # print(x1.shape)
        # print(x2.shape)
        # return x1 + x2
        return self.weight[None, :, None, None] * tmp + self.bias[None, :, None, None]

    def forward_1d(self, x):
        tmp = (x - self.mean) / np.sqrt(self.var)
        # x1 = self.weight[None, :, None, None] * tmp
        # x2 = self.bias[None, :, None, None]
        # print(x1.shape)
        # print(x2.shape)
        # return x1 + x2
        return self.weight * tmp + self.bias


class Linear:
    def __init__(self, weight, bias) -> None:
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        # print(x.shape, self.weight.T.shape, self.bias.shape)
        # _s = x.shape
        # x.
        # return np.matmul(x.reshape(x.size, 1), self.weight.T) + self.bias
        return np.matmul(x, self.weight.T) + self.bias


class Net:
    def __init__(self) -> None:


        self.conv1 = Conv2d(weight=np.array(data['conv1.weight']), bias=np.array(data['conv1.bias']), stride=1, padding=1)
        self.conv2 = Conv2d(weight=np.array(data['conv2.weight']), bias=np.array(data['conv2.bias']), stride=1, padding=1)
        self.conv3 = Conv2d(weight=np.array(data['conv3.weight']), bias=np.array(data['conv3.bias']), stride=1, padding=0)

        self.bn1 = BatchNorm(weight=np.array(data['bn1.weight']), bias=np.array(data['bn1.bias']),
                             mean=np.array(data['bn1.running_mean']),
                             var=np.array(data['bn1.running_var']))
        self.bn2 = BatchNorm(weight=np.array(data['bn2.weight']), bias=np.array(data['bn2.bias']),
                             mean=np.array(data['bn2.running_mean']),
                             var=np.array(data['bn2.running_var']))
        self.bn3 = BatchNorm(weight=np.array(data['bn3.weight']), bias=np.array(data['bn3.bias']),
                             mean=np.array(data['bn3.running_mean']),
                             var=np.array(data['bn3.running_var']))

        self.fc1 = Linear(weight=np.array(data['fc1.weight']), bias=np.array(data['fc1.bias']))
        self.fc_bn1 = BatchNorm(weight=np.array(data['fc_bn1.weight']), bias=np.array(data['fc_bn1.bias']),
                                mean=np.array(data['fc_bn1.running_mean']), var=np.array(data['fc_bn1.running_var']))

        self.fc2 = Linear(weight=np.array(data['fc2.weight']), bias=np.array(data['fc2.bias']))
        self.fc_bn2 = BatchNorm(weight=np.array(data['fc_bn2.weight']), bias=np.array(data['fc_bn2.bias']),
                                mean=np.array(data['fc_bn2.running_mean']), var=np.array(data['fc_bn2.running_var']))

        self.fc3 = Linear(weight=np.array(data['fc3.weight']), bias=np.array(data['fc3.bias']))
        self.fc4 = Linear(weight=np.array(data['fc4.weight']), bias=np.array(data['fc4.bias']))

    def predict(self, board):
        x = board.view().astype(np.float32).reshape(1, 1, 8, 8)
        x = relu(self.bn1.forward(self.conv1.forward(x)))
        x = relu(self.bn2.forward(self.conv2.forward(x)))
        x = relu(self.bn3.forward(self.conv3.forward(x)))
        x = x.reshape(-1, 144)
        x = relu(self.fc_bn1.forward_1d(self.fc1.forward(x)))
        x = relu(self.fc_bn2.forward_1d(self.fc2.forward(x)))
        pi = self.fc3.forward(x)
        v = self.fc4.forward(x)
        return np.exp(log_softmax(pi))[0], np.tanh(v)


import math


EPS = 1e-8

BLACK, BLANK, WHITE = -1, 0, 1
directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


class Board:

    def __init__(self, n: int, board = None) -> None:
        self._size: int = n
        if board is None:
            self._pieces = np.zeros((n, n), dtype=np.int8)
            self._pieces[(n // 2 - 1, n // 2)] = WHITE
            self._pieces[(n // 2, n // 2 - 1)] = WHITE
            self._pieces[(n // 2 - 1, n // 2 - 1)] = BLACK
            self._pieces[(n // 2, n // 2)] = BLACK
        else:
            self._pieces = board

    @property
    def board(self):
        return self._pieces

    def __getitem__(self, item):

        return self._pieces.item(tuple(item))

    def __str__(self) -> str:
        ans = ''
        for index, i in enumerate(np.nditer(self._pieces, order='C')):
            if not index % self._size:
                ans += '\n'
            ans += f"{i} "
        return ans

    def count_score(self, color: int) -> int:
        return -color * self._pieces.sum()


    
    def get_legal_moves(self, color: int) -> Set[Tuple[int, int]]:
        ans: Set[Tuple[int, int]] = set()
        search_list: List[Tuple[int, int]] = list(zip(*np.where(self._pieces == color)))
        for i in search_list:
            for j in directions:
                new = self._check_flip(i, j, color)
                if new is not None:
                    ans.add(new)
        return ans

    def can_move(self, color: int) -> bool:
        search_list: List[Tuple[int, int]] = list(zip(*np.where(self._pieces == color)))
        if not search_list:
            return False
        for i in search_list:
            for j in directions:
                if self._check_flip(i, j, color):
                    return True
        return False

    
    def _check_flip(self, origin: Tuple[int, int], direction: Tuple[int, int], color: int) -> \
            Optional[Tuple[int, int]]:
        _get = False
        for pos in Board._move(self._size, origin, direction):
            if self[pos] == 0:
                return pos if _get else None
            elif self[pos] == color:
                return None
            else:
                _get = True
        return None


    def move(self, position: Tuple[int, int], color: int) -> bool:
        flips = tuple(zip(*set(i for direction in directions for i in self._get_flips(position, direction, color))))
        if not flips:
            return False
        self._pieces[flips] = color
        return True

    def _get_flips(self, origin: Tuple[int, int], direction: Tuple[int, int], color: int) -> \
            List[Tuple[int, int]]:
        _get = [origin]
        for pos in Board._move(self._size, origin, direction):
            if self[pos] == 0:
                return []
            if self[pos] == -color:
                _get.append(pos)
            elif self[pos] == color and _get:
                return _get
        return []

    @staticmethod
    def _move(n: int, origin: Tuple[int, int], direction: Tuple[int, int]) -> Tuple[int, int]:
        pos = list(map(sum, zip(origin, direction)))
        while all(map(lambda _p: 0 <= _p < n, pos)):
            yield tuple(pos)
            pos = list(map(sum, zip(pos, direction)))


square_content = {
    -1: "X",
    +0: "-",
    +1: "O"
}


class Game:

    def __init__(self, n: int) -> None:
        self._size = n
        self._action_size = n ** 2 + 1

    def getInitBoard(self):
        return Board(self._size).board

    def getBoardSize(self) -> Tuple[int, int]:
        return tuple([self._size, self._size])

    def get_action_size(self) -> int:
        return self._action_size

    def getNextState(self, board, player: int, action: int):
        if action == self._action_size - 1:
            return tuple([board, -player])
        tmp = Board(self._size, board.copy())
        tmp.move((action // self._size, action % self._size), player)
        return tuple([tmp.board, -player])

    def getValidMoves(self, board, player: int):
        valid = [0] * self._action_size
        tmp = Board(self._size, board.copy())
        moves = tmp.get_legal_moves(player)
        if not moves:
            valid[-1] = 1
            return valid
        for x, y in moves:
            valid[self._size * x + y] = 1
        return np.array(valid)

    def is_ended(self, board, player: int) -> int:
        tmp = Board(self._size, board.copy())
        if tmp.can_move(player) or tmp.can_move(-player):
            return 0
        return 1 if tmp.count_score(player) > 0 else -1

    def getCanonicalForm(self, board, player: int):
        return player * board

    def getSymmetries(self, board, pi) -> list:
        pi_board = np.reshape(pi[: -1], (self._size, self._size))
        l = []
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def HASH(self, board) -> str:
        return board.tostring()

    def HASHReadable(self, board) -> str:
        board_s = "".join(square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player: int) -> int:
        tmp = Board(self._size, board.copy())
        return tmp.count_score(player)



class MCTS():

    def __init__(self, game, nnet):
        self.cpuct = 1.2
        self.game = game
        self.nnet = nnet
        self.Qsa = {}  
        self.Nsa = {}  
        self.Ns = {}  
        self.Ps = {}  
        self.Es = {}  
        self.Vs = {}

    def getActionProb(self, canonicalBoard, temp=1):
        start = time.time()
        while time.time() - start < 4.1:
            self.search(canonicalBoard)
        s = self.game.HASH(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_size())]
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA, probs = np.random.choice(bestAs), [0] * len(counts)
            probs[bestA] = 1
            return probs
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        s = self.game.HASH(canonicalBoard)
        if s not in self.Es:
            self.Es[s] = self.game.is_ended(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
        valids, cur_best, best_act = self.Vs[s], -float('inf'), -1
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                if u > cur_best:
                    cur_best = u
                    best_act = a
        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        v = self.search(next_s)
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return -v


class AI:
    def __init__(self, chessboard_size: int, color: int, time_out: int) -> None:
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.game = Game(8)
        self.net = Net()
        self.mcts = MCTS(self.game, self.net)


    def go(self, chessboard):
        # print(f"current color is {self.color}")

        self.candidate_list.clear()
        actions = Board(8, chessboard).get_legal_moves(self.color)
        self.candidate_list = list(actions)
        if not actions or len(actions) == 1:
            return self.candidate_list[-1]
        chessboard = self.game.getCanonicalForm(chessboard, self.color)
        # self.game.display(chessboard)

        prob = self.mcts.getActionProb(chessboard)
        action = np.argmax(prob)
        x, y = action // 8, action % 8
        self.candidate_list.append((x, y))
        return self.candidate_list[-1]


# if __name__ == '__main__':
    # self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
    # self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
    # self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
    # data = torch.load('temp/best.pth.tar', map_location='cpu')['state_dict']
    #
    # conv1 = Conv2d(weight=data['conv1.weight'].numpy(), bias=data['conv1.bias'].numpy(), stride=1, padding=1)
    # conv2 = Conv2d(weight=data['conv2.weight'].numpy(), bias=data['conv2.bias'].numpy(), stride=1, padding=1)
    # conv3 = Conv2d(weight=data['conv3.weight'].numpy(), bias=data['conv3.bias'].numpy(), stride=1, padding=1)
    #
    # bn1 = BatchNorm(weight=data['bn1.weight'].numpy(), bias=data['bn1.bias'], mean=data['bn1.running_mean'].numpy(),
    #                 var=data['bn1.running_var'].numpy())
    # bn2 = BatchNorm(weight=data['bn2.weight'].numpy(), bias=data['bn2.bias'], mean=data['bn2.running_mean'].numpy(),
    #                 var=data['bn2.running_var'].numpy())
    # bn3 = BatchNorm(weight=data['bn3.weight'].numpy(), bias=data['bn3.bias'], mean=data['bn3.running_mean'].numpy(),
    #                 var=data['bn3.running_var'].numpy())
    #
    # fc1 = Linear(weight=data['fc1.weight'].numpy(), bias=data['fc1.bias'].numpy())
    # fc_bn1 = BatchNorm(weight=data['fc_bn1.weight'].numpy(), bias=data['fc_bn1.bias'],
    #                    mean=data['fc_bn1.running_mean'].numpy(), var=data['fc_bn1.running_var'].numpy())
    #
    # fc2 = Linear(weight=data['fc2.weight'].numpy(), bias=data['fc2.bias'].numpy())
    # fc_bn2 = BatchNorm(weight=data['fc_bn2.weight'].numpy(), bias=data['fc_bn2.bias'],
    #                    mean=data['fc_bn2.running_mean'].numpy(), var=data['fc_bn2.running_var'].numpy())
    #
    # fc3 = Linear(weight=data['fc3.weight'].numpy(), bias=data['fc3.bias'].numpy())
    # fc4 = Linear(weight=data['fc4.weight'].numpy(), bias=data['fc4.bias'].numpy())
    #
    # inp = np.ones((1, 1, 8, 8), dtype=float)

    # x1 = np.array([1,2,3])
    # x2 = np.array([1,2,3])
    # print(np.matmul(x1, x2.T))
    # print(x1 + x2)

    # board: NDArray = np.zeros((8, 8), dtype=np.int8)
    # board[(8 // 2 - 1, 8 // 2)] = 1
    # board[(8 // 2, 8 // 2 - 1)] = 1
    # board[(8 // 2 - 1, 8 // 2 - 1)] = -1
    # board[(8 // 2, 8 // 2)] = -1
    # board = board.astype(np.float32).reshape(1, 1, 8, 8)
    # net = Net()
    # net.predict(board)
    # 45   53  62  32
    # g = Game(8)
    # player = 1
    # board = g.getInitBoard()
    # nnet = Net()
    # # nnet.load_checkpoint('/home/satan/桌面/CS303/temp', 'best.pth.tar')
    # mcts = MCTS(g, nnet)
    # ai = AI(8, -1, 5)
    #
    # human = GreedyPlayer(g)
    # while True:
    #     g.display(board)
    #     print('')
    #     if player < 0:
    #         start = time.time()
    #         # print(board.shape)
    #         act = ai.go(board)
    #         # acts = mcts.getActionProb(board)
    #         # for i, j in enumerate(acts):
    #         #     if j > 0:
    #         #         print(i // 8, i % 8)
    #         # index = np.argmax(acts)
    #         end = time.time()
    #
    #         print(f"MCTS Time: {(end - start)}")
    #         # print(g.getScore(board, 1))
    #         board, player = g.getNextState(board, player, act)
    #         print(f"\n----------------------\n{act // 8}, {act % 8} \n ---------------------\n")
    #     else:
    #         # _ = input('\nHuman Play')
    #         # board, player = g.getNextState(board, player, human.play(board, player))
    #         # print(g.getScore(board, -1))
    #         print(Board(8, board).get_legal_moves(player))
    #         x, y = input().split()
    #         board, player = g.getNextState(board, player, int(x) * 8 + int(y))
    #     ended = g.is_ended(board, player)
    #     if ended != 0:
    #         if ended == 1:
    #             print(f"Ended! {player} Win")
    #         else:
    #             print(f"Ended! {-player} Win")
    #         break
    #