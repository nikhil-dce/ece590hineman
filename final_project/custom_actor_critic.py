from gym.spaces import Box, Discrete
import torch
import torch.nn as nn

# import spinningup.spinup.algos.pytorch.vpg.core
# from spinup.algos.pytorch.vpg.core import MLPCritic, Actor
from spinup.algos.pytorch.vpg.core import *

import numpy as np

class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape
        act_dim = action_space.shape
        print ("This is the obs_dim: " + str(obs_dim[0]))
        print ("This is the action space: ", str(act_dim[0]))
        print ("Hidden sizes: ", hidden_sizes)

        # policy builder depends on action space
        # if isinstance(action_space, Box):
        self.pi = MLPGaussianActor(obs_dim[0], act_dim[0], hidden_sizes, activation)
        # elif isinstance(action_space, Discrete):
        # self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim[0], hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]