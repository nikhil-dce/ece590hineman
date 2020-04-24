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

class BayesianMLPActorCritic(nn.Module):
    
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape
        act_dim = action_space.shape
        
        self.pi = MLPGaussianActor(obs_dim[0], act_dim[0], hidden_sizes, activation)
        self.v  = BayesianMLPCritic(obs_dim[0], hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class BayesianMLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = BayesianMLP([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


def BayesianMLP(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)