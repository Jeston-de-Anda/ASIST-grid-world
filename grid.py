import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

class State:
    beta = 1.
    
    mask = 65535
    
    @staticmethod
    def to_token(a, b):
        """
        a and b: are int(s) or numpy arrays of dtype=int of same shape
        """
        return np.array(a*State.mask + b, dtype=int)
    
    @staticmethod
    def to_coord(token):
        
        return np.array([token//State.mask, token%State.mask], dtype=int).T
    
    @classmethod
    def set_beta(cls, beta):
        cls.beta = beta
    
    @classmethod
    def set_agents(cls,targets=None, agents=None, n_row=10, n_col=10):
        """
        Parameters
        ----------
        targets: iterable of targets' original coordinates from (1,1) to (n,m)
        agents:  iterable of agents' original coordinates
        """
        cls.n_row = n_row
        cls.n_col = n_col
        if targets is None:
            cls.targets = np.array([State.to_token(np.random.randint(0,cls.n_row),np.random.randint(0,cls.n_col))])
        else:
            cls.targets = State.to_token(*np.array(targets).T)
        
        if agents is None:
            cls.agents = np.array([State.to_token(np.random.randint(0,cls.n_row),np.random.randint(0,cls.n_col))])
        else:
            cls.agents = State.to_token(*np.array(agents).T)
        
    def __init__(self, i, j):
        self.coord = np.array([i, j])
        self.token = State.to_token(i, j)
        
        self.Q = []
        self.value = 1.
        self.action = []
        self.cost = []
        # self.map = {}
        
    def add_action(self, action, cost):
        self.action += [action,]
        self.cost += [cost,]
        self.Q += [0,]
        
        # self.map[action] = cost
        
    def change_cost(self, action, cost):
        self.cost[self.action == action] = cost
        # self.map[action] = cost
        
        
    def update_Q(self):
        for i in range(len(self.action)):
            self.Q[i] = self.action[i].value + self.cost[i]

    def update_V(self, targets=None, values=None):
        '''
        targets: self-defined targets, iterable, in terms of token
        values: dict, key=target token, value=value of target
        '''
        if targets is None:
            targets = self.__class__.targets
        if self.token in targets:
            self.value = values[self.token] if values is not None else 0
            return self.value
        old_value = self.value
        prob = np.exp(np.array(self.Q) * State.beta)
        prob /= np.sum(prob)
        self.value = np.matmul(prob, np.array(self.Q))
        return abs(self.value - old_value)



def set_board(n=10, m=10, blocks=[],targets=None, agents=None,beta=1.0):
    
    
    State.set_beta(beta)
    State.set_agents(n_row=n, n_col=m,targets=targets, agents=agents)


    seq = [None] * ((m+2)*(n+2))

    seq = np.array(seq, dtype=State)
    board = seq.reshape(n+2,m+2)

    for i in range(m*n):
        board[i//m+1, i%m+1] = State(i//m, i%m)

    for block in blocks:
        # assert isinstance(block[0],int)
        board[int(block[0]), int(block[1])] = None
        
        
    directions = [np.array([-1,0]),
                  np.array([1,0]),
                  np.array([0,-1]),
                  np.array([0,1])]
    for i in range(1,n+1):
        for j in range(1, m+1):
            if board[i,j] is None:
                continue
            for d in directions:
                if board[i+d[0], j+d[1]] is not None:
                    board[i,j].add_action(board[i+d[0], j+d[1]], 0 if State.to_token(i+d[0], j+d[1]) in State.targets else -1)

    return seq, board


def update(board, seq=None, targets=None, values=None):
    '''
    could be more flexible, but less efficient
    '''
    diff = np.zeros_like(seq, dtype=float)
    for element in seq:
        if element is not None:
            element.update_Q()
    for i, element in enumerate(seq):
        if element is not None:
            diff[i] = element.update_V(targets, values)
    
    return diff