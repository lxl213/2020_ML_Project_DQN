
import numpy as np
from collections import namedtuple, deque

import random
import pytrace
from pytrace import optim
import pytrace.nn as nn
from models import myDQN_network

class Agent:
    
    def __init__(self):
        
        self.qnetwork_local = myDQN_network()
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if not isinstance(state, np.ndarray):
            state = np.asarray(state).astype('float')
        state = pytrace.from_numpy(state).float().unsqueeze(0)
        action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))