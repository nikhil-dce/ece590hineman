from gym.spaces import Box, Discrete
import torch
import torch.nn as nn

# import spinningup.spinup.algos.pytorch.vpg.core
# from spinup.algos.pytorch.vpg.core import MLPCritic, Actor
from spinup.algos.pytorch.vpg.core import *

import numpy as np
from algos.vpg_source.bnn import BayesLinear

import math

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

class BayesMLPActorCritic(nn.Module):
    
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape
        act_dim = action_space.shape
        
        self.pi = MLPGaussianActor(obs_dim[0], act_dim[0], hidden_sizes, activation)
        self.v  = BayesMLPCritic(obs_dim[0], hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class BayesMLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = BayesMLP([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

    def compute_kl(self):
        
        device = torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")
        kl = torch.Tensor([0]).to(device)
        kl_sum = torch.Tensor([0]).to(device)
        # n = torch.Tensor([0]).to(device)

        for m in self.v_net.children():

            if isinstance(m, BayesLinear):
                kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl
                # n += len(m.weight_mu.view(-1))

                if m.bias :
                    kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma)
                    kl_sum += kl
                    # n += len(m.bias_mu.view(-1))
        
        return kl_sum


def BayesMLP(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [BayesLinear(0.0, 1.0, sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1) :
    """
    An method for calculating KL divergence between two Normal distribtuion.
    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).

    """
    kl = log_sigma_1 - log_sigma_0 + \
    (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5
    return kl.sum()