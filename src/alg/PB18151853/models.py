
import torch
from pytrace.nn import Model
import pytrace.nn as nn


class myDQN_network(nn.Model):
    def __init__(self):
        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
                                  nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()).set_name("Conv")
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(), nn.Linear(512, 18)).set_name("Full-connect")
        super(myDQN_network, self).__init__()
        self.set_name('DQN')

    def construct(self):
        return [self.conv, self.fc]
    
    def forward(self, obs):
        obs = self.conv(obs)
        obs = obs.View(obs.shape[0], obs.shape[1] * obs.shape[2] * obs.shape[3])
        actions = self.fc(obs)
        return actions