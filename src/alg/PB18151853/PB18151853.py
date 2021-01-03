import sys
root_path = './src/alg/PB18151853'
sys.path.append(root_path)
sys.path.append("./src/alg") # import pytrace

from os.path import join
from copy import deepcopy
from gym.spaces import Discrete

from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file


import pytrace 
from pytrace import tracer
from agent import Agent
import cv2
import numpy as np
from collections import deque

# ----------------------------------------------------------
# set random seed to re-perform
seed = 0
pytrace.set_seed(seed)

class PB18151853(RL_alg):
    def __init__(self,ob_space,ac_space):
        super().__init__()
        assert isinstance(ac_space, Discrete)

        self.team = ['PB17121707','PB17121732','PB18151853'] # 记录队员学号
        # self.config = get_params_from_file('src.alg.PB00000000.rl_configs',params_name='params') # 传入参数
        
        self.ac_space = ac_space
        self.state_dim = ob_space.shape
        print(self.state_dim)
        self.action_dim = ac_space.n
        # ----------------------------------------------------------
        # initialize implemented DQN models and weight
        self.agent  = Agent()
        # details about the api of Model in pytrace.nn.Model
        self.agent.qnetwork_local.load_seq_list(
            pytrace.load(
                join(root_path, './riverraid/best_list.pth')
                )
            )
        pytrace.prYellow(f"load weights from: {join(root_path, './riverraid/best_list.pth')}")
        self.state = np.zeros([4, 84, 84])
        #self.state = self.WarpFrame(self.state)
        #self.state = np.stack([self.state] * 4, axis=0)
        # self.state = deque([np.zeros([84, 84, 4])], maxlen=4)
        
        
    def step(self, state):
        self.state = self.FrameStack(state, self.state)
        action = self.agent.act(self.state)
        return action

    def explore(self, obs):
        raise NotImplementedError

    def test(self):
        print('??')

    def WarpFrame(self, obs):
        """
        :param obs: The raw observation returned by env, it should be a (210 * 160 * 3) RGB frame
        :return: ans: A (84 * 84) compressed gray style frame normalized in [0, 1]
        """
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        #return frame[:, :, None]
        return frame / 255.0

    def FrameStack(self, new_obs, obs):
        """
        :param new_obs: A raw observation returned by env, it should be a (210 * 160 * 3) RGB frame
        :param obs: The stack of past 4 (84 * 84) compressed gray style frames
        :return: A new stack of past 4 (84 * 84) compressed gray style frames
        """
        new_obs = self.WarpFrame(new_obs)
        obs[0 : 3, :, :] = obs[1 :, :, :]
        obs[3, :, :] = new_obs
        return obs

