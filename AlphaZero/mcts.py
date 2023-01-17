import numpy as np
from src_net import get_next_state, check, get_candidate, get_player


class Node:
    def __init__(self, parent, prob, state, action, state_now=None, c_puct=5):
        self.N, self.W, self.Q, self.P = 0, 0, 0, prob
        self.parent, self.children = parent, {}
        self.state, self.action, self.state_now = state, action, state_now
        self.c_puct = c_puct
    
    def get_value(self):
        return self.Q+self.c_puct*self.P*np.sqrt(self.parent.N)/(1+self.N)


class MCT:
    def __init__(self, root, policy_value_fn, c_puct=5):
        self.root = root
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
    
    def is_root(self, n):
        return self.root==n
    

    def select(self, n):
        return max(n.children.values(),key=lambda node: node.get_value())

    def expand(self, n):
        if(n.state_now is None):
            n.state_now = get_next_state(n.state, n.action)
        actions = get_candidate(n.state_now)
        probs = self.policy_value_fn(n.state_now)[0]
 
        n.children = {action:Node(n, probs[action], n.state_now, action, c_puct=self.c_puct) for action in actions}  
    
    def backup(self, n):
        if(n.state_now is None):
            n.state_now = get_next_state(n.state, n.action)
        end, player = check(n.state_now)
        if(not end):
            player = get_player(n.state_now)
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
            end, winner = check(cur.state_now)
            if(not end):
                self.expand(cur)
                cur = self.select(cur)
   
            self.backup(cur)    
    
    def move(self, action):
        if(not self.root.children):
            self.expand(self.root)
        self.root = self.root.children[action]
        self.root.parent = None
    
    def play(self, temp=1):
        actions_probs = [(action, np.float_power(node.N, 1.0/temp)) for action, node in self.root.children.items()]
        actions, probs = zip(*actions_probs)

        # print(actions_probs)
        probs /= np.sum(probs)
        action = actions[np.random.choice(len(actions), p=probs)]
        self.move(action)
        return action