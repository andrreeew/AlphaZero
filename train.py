import torch
import torch.nn as nn
import numpy as np
import random
from mcts import MCT, Node
from net import PolicyValueNet
from src_net import init_state
from collections import deque
import time
from compete import compete_net
from torch.utils.data import Dataset, DataLoader



class MyDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor(np.array([d[0] for d in data])).float()
        self.y = [(torch.tensor(np.array(d[1])).float(), torch.tensor(np.array(d[2])).float()) 
                                for d in data]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return len(self.y)
    


def train_epoch(net, train_iter, loss, updater):
    net.train()

    policy_loss_sum, value_loss_sum, cnt = 0, 0, 0
    for X, y in train_iter:
        y_pred = net(X)
        policy_loss, value_loss = loss(y_pred, y)
        l = policy_loss+value_loss
        updater.zero_grad()
        l.mean().backward()
        updater.step()

        policy_loss_sum += float(policy_loss.sum())
        value_loss_sum += float(value_loss.sum())
        cnt += y_pred[0].shape[0]

    return policy_loss_sum/cnt, value_loss_sum/cnt


def train(net, loss, updater, data, batch_size, num_epochs, sample_size):
    train_dataset = MyDataset(random.sample(data, sample_size))
    train_iter = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    net.train()

    for epoch in range(num_epochs):
        policy_loss, value_loss = train_epoch(net, train_iter, loss, updater)
      
    return policy_loss, value_loss


def preprocess_data(data):
    size = data[0][0].shape[-1]
    for i in range(len(data)):
        actions, probs = data[i][1]
        action_prob = np.zeros((size, size), dtype=float)
        for j in range(len(actions)):
            action_prob[actions[j]] = probs[j]
        data[i][1] = action_prob

    augment_data = []
    for state, action_prob, v in data:
        for i in [1, 2, 3, 4]:
            equal_state = np.rot90(state, i, axes=(-2,-1))
            equal_action_prob = np.rot90(action_prob, i, axes=(-2, -1))

            augment_data.append([equal_state, equal_action_prob, v])
            augment_data.append([np.flip(equal_state, axis=(-1)), np.flip(equal_action_prob, axis=(-1)), v])        
    return augment_data



class PolicyValueLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_loss = nn.CrossEntropyLoss(reduction='none')
        self.value_loss = nn.MSELoss(reduction='none')
    
    def forward(self, y_pred, y):
        batch_size = y_pred[0].shape[0]
        policy_loss = self.policy_loss(y_pred[0].reshape(batch_size, -1), y[0].reshape(batch_size, -1))
        value_loss = self.value_loss(y_pred[1], y[1])
        return policy_loss, value_loss

    
def save(net, data_buffer, chessboard_size, game_name):
    torch.save(data_buffer, game_name+str(chessboard_size)+'_data_buffer.data')
    torch.save(net.state_dict(), game_name+str(chessboard_size)+'_tmp.params')


if __name__ == '__main__':
    CHESSBOARD_SIZE = 6
    GAME = 'reversi'

    buffer_size = 32*8*500
    data_buffer = deque(maxlen=buffer_size)
    batch_size = 1024
    num_epochs = 5
    num_games = 1
    checkpoint = 100
    def sample_size():
        return batch_size

    c_cput = 2
    num_simulate = 50
    lr = 2e-3

    net = PolicyValueNet(CHESSBOARD_SIZE)
    policy_value_loss = PolicyValueLoss()



    try:
        data_buffer.extend(torch.load(GAME+str(CHESSBOARD_SIZE)+'_data_buffer.data'))
        net.load_state_dict(torch.load(GAME+str(CHESSBOARD_SIZE)+'_tmp.params'))
        print('load existed model')
    except:
        print('no existed model, train from zero.')
        torch.save(net.state_dict(), GAME+str(CHESSBOARD_SIZE)+'.params')

    cnt = 0
    loss_sum, policy_loss_sum = 0, 0
    while(True):
        cnt += 1
        mct = MCT(policy_value_fn=net, c_puct=c_cput)
        start = time.time()
        game_cnt = 0
        while(game_cnt<num_games):
            mct.set_root(Node(state_now=init_state(CHESSBOARD_SIZE)))
            data_buffer.extend(preprocess_data(mct.self_play(num=num_simulate)))
            game_cnt += 1
            if(game_cnt%10==0 and game_cnt!=num_games):
                print(game_cnt, end=' ')
        print('step:', cnt, 'game_num:', game_cnt)


        updater = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        # updater = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        if(len(data_buffer)<sample_size()):
            continue
        policy_loss, value_loss = train(net, policy_value_loss, updater, data_buffer, batch_size, num_epochs, 
                                            sample_size=sample_size())
        
        loss_sum += policy_loss+value_loss
        policy_loss_sum += policy_loss

        
        print('step:', cnt, 'cost time:', time.time()-start, 
                                'loss:', policy_loss+value_loss, 'entropy loss:', policy_loss)
        
        if(cnt%checkpoint==0):
            save(net, data_buffer, CHESSBOARD_SIZE, GAME)

            print('step', cnt-checkpoint+1, 'to', cnt, end=' ')
            print('average loss:', loss_sum/checkpoint, 'average entropy loss:', policy_loss_sum/checkpoint)
            loss_sum, policy_loss_sum = 0, 0

            net_old = PolicyValueNet(CHESSBOARD_SIZE)
            net_old.load_state_dict(torch.load(GAME+str(CHESSBOARD_SIZE)+'.params'))
            print('compete with old net', end=' ')
            cnt1, cnt2 = compete_net(net, net_old, CHESSBOARD_SIZE, 30)
            print('win:', cnt1, 'lose:', cnt2)
            if(cnt1+0.5*(30-cnt1-cnt2)>30*0.6):
                torch.save(net.state_dict(), GAME+str(CHESSBOARD_SIZE)+'.params')
                print('new net update!!!')
 