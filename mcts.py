import numpy as np
import torch
import torch.nn as nn
from src_net import * 

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class Node:
    def __init__(self, parent=None, prob=1, state=None, action=None, state_now=None, c_puct=5):
        self.N, self.W, self.Q, self.P = 0, 0, 0, prob
        self.parent, self.children = parent, {}
        self.state, self.action, self.state_now = state, action, state_now
        self.end, self.winner = None, None
        self.c_puct = c_puct
    
    def check(self):
        if(self.end is not None):
            return self.end, self.winner
        if(self.state_now is None):
            self.state_now = get_next_state(self.state, self.action)
        self.end, self.winner = check(self.state_now)
        return self.end, self.winner

    def get_value(self):
        return self.Q+self.c_puct*self.P*np.sqrt(self.parent.N)/(1+self.N)


class MCT:
    def __init__(self, root=None, policy_value_fn=None, c_puct=2):
        self.root = root
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct

    def set_root(self, root):
        self.root = root
    
    def is_root(self, n):
        return self.root==n
    

    def select(self, n):
        return max(n.children.values(),key=lambda node: node.get_value())

    def expand(self, n):
        if(n.state_now is None):
            n.state_now = get_next_state(n.state, n.action)
        actions = get_candidate(n.state_now)
        if(isinstance(self.policy_value_fn, nn.Module)):
            probs = self.policy_value_fn(torch.tensor(n.state_now).float().unsqueeze(0))[0].detach().numpy()
        else:
            probs = self.policy_value_fn(n.state_now)[0]
 
        n.children = {action:Node(n, probs[action], n.state_now, action, c_puct=self.c_puct) for action in actions}  
    
    def backup(self, n):
        if(n.state_now is None):
            n.state_now = get_next_state(n.state, n.action)
        end, player = n.check()
        if(not end):
            player = get_player(n.state_now)
            if(isinstance(self.policy_value_fn, nn.Module)):
                v = self.policy_value_fn(torch.tensor(n.state_now).float().unsqueeze(0))[1].detach().numpy()
            else:
                v = self.policy_value_fn(n.state_now)[1]
         
        else:
            v = 1

        cur = n
        while(not self.is_root(cur)):
            cur.N += 1
            if(player==get_player(cur.state)):
                cur.W += v
            elif(player==-get_player(cur.state)):
                cur.W += -v
            cur.Q = 1.0*cur.W/cur.N
            cur = cur.parent
        cur.N += 1

            

    def simulate(self, num=1):
        for _ in range(num):
            cur = self.root
            while(cur.children):
                cur = self.select(cur)
            
            if(cur.state_now is None):
                cur.state_now = get_next_state(cur.state, cur.action)
            end, winner = cur.check()
            if(not end):
                self.expand(cur)
                cur = self.select(cur)
   
            self.backup(cur)    
    
    def move(self, action):
        if(not self.root.children):
            self.expand(self.root)
        self.root = self.root.children[action]
        self.root.parent = None
    

    def get_current_state(self):
        if(self.root.state_now is None):
            self.root.state_now = get_next_state(self.root.state,self.root.action)
        
        return self.root.state_now
    
    def get_actions_probs(self, temp=1e-3):
        actions_visits = [(action, node.N) for action, node in self.root.children.items()]
        actions, visits = zip(*actions_visits)
        # print(actions_probs)
        if(temp==1):
            sum = np.sum(np.array(visits))
            probs = np.array(visits)/sum
        else:
            probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return actions, probs
    
    def play(self, temp=1e-3):
        actions, probs = self.get_actions_probs(temp=temp)
        action = actions[np.random.choice(len(actions), p=probs)]
        # print(actions, probs)
        self.move(action)
        return action
    
    #自己与自己下生成数据
    def self_play(self, num=100, temp=1):
        data = []
        cnt = 0
        while(True):
            cnt += 1
            self.simulate(num)
            if(get_left_step(self.get_current_state())<cnt):
                temp = 0.001 
            actions, probs = self.get_actions_probs(temp)
            data.append([self.get_current_state(), (actions, probs)])
            action = actions[np.random.choice(len(actions), 
                                p=0.75*probs+0.25*np.random.dirichlet(0.03*np.ones(len(probs))))]
            self.move(action)

            end, winner = self.root.check()
            if(end):
                break
        
        for i in range(len(data)):
            if(get_player(data[i][0])==winner):
                data[i].append(1)
            elif(get_player(data[i][0])==-winner):
                data[i].append(-1)
            else:
                data[i].append(0)
        
        return data
    
