import sys
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(precision=4)

import gym
from gym import wrappers, logger
# from gridworld_env import gridworld

from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from agents import RandomAgent

import torch

from multigoal_env.multigoal import MultiGoal

env_id = 'MultiGoal-v0'
seed = 1234

env = gym.make(env_id)
env.seed(seed)
agent = RandomAgent(env.action_space)
reward = 0.
done = False

alpha = 0.01
gamma = 0.9

def process_sample(observation, action,reward, 
                    terminal,next_observation,info):
    
    processed_observation = {
        'observations': observation,
        'actions': action,
        'rewards': np.atleast_1d(reward),
        'terminals': np.atleast_1d(terminal),
        'next_observations': next_observation,
        'infos': info,
    }

    return processed_observation


current_path = []
path_length = 0
path_return = 0.
num_episodes = 1

for episode in range(1, num_episodes+1):
    
    current_ob = env.reset()
    rewards = []
    
    while True:
        action = agent.act(current_ob, reward, done)
        ob, reward, done, info = env.step(action)

        processed_sample = process_sample(observation=current_ob, action=action, reward=reward,
                                          terminal=done, next_observation=ob,info=info)
        
        current_path.append(processed_sample)
        
        env.render_rollouts(current_path)
        path_length += 1

        if done or path_length > 50:
            current_ob = env.reset()
            break
        else:
            current_ob = ob
            
#             last_path = tree.map_structure(lambda *x: np.stack(x, axis=0), *self._current_path)
#             pass
#         s_old = s_new
        # Note there's no env.render() here. But the environment still can open window and
        # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
        # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        
        
        
    if episode % 5 == 0:
        print ('Episode Number: {:d} | Total Reward: {:.4f} '.format(episode, np.sum(rewards)))

# Close the env and write monitor result info to disk
env.close()