"""
Gridworld Environment
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class GridWorld(gym.Env):
	"""
	Description:
		Gridworld environment as specified in Sutton et al. Example 3.5
	Observation: 
		Discrete (width*height)
	Actions:
		Discrete (4)
		North, South, West, East
	Reward:
		Reward as specified in Example 3.5 of Sutton et al.
		The reward matrix is specified in __init__.py
	Starting State:
		Starting state is a location on the grid
	Episode Termination:
		Episode terminates after MAX_STEPS_PER_EPISODE
	"""
	
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	ACTION = ["N", "S", "E", "W"]
	MAX_STEPS_PER_EPISODE = 20

	def __init__(self, reward_grid, jump_transitions=None):
		
		assert len(reward_grid.shape) == 2, "Shape should be 2D"

		self.reward_grid = reward_grid
		self.w, self.h = reward_grid.shape

		if jump_transitions is None:
			jump_transitions  = dict()
		
		self.action_space = spaces.Discrete(4)
		self.observation_space = spaces.Discrete(self.w*self.h)
		self.jump_transitions = jump_transitions

		self.all_state = np.eye(self.observation_space.n)
		self.seed(seed=1234)
		self.reset()
	
	def reset(self):
								
		# define x_random, y_random
		ob = self.observation_space.sample()
		
		# y_random = ob // self.w
		# x_random = ob - y_random*self.w
		x_random = ob // self.w
		y_random = ob - x_random*self.w

		self.position = (x_random, y_random)
		
		# self.state = np.zeros((self.reward_grid.shape))
		# self.state[x_random, y_random] = 1

		# Not done
		self.done = False

		# step_count
		self.step_count = 0

		return self.all_state[x_random*self.w + y_random]
		# return self.position
		# return self.state

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

		self.step_count += 1	

		# state = self.state
		action = self.ACTION[action]
		
		x_prev, y_prev = self.position[0], self.position[1]
		x_curr, y_curr = x_prev, y_prev
		# state[x_prev, y_prev] = 0
		
		if (x_prev, y_prev) in self.jump_transitions:
			data = self.jump_transitions[(x_prev, y_prev)]
			x_curr, y_curr = data
			# x_curr, y_curr = data['next_state']
			# reward = data['reward']
		else:
			if action == 'N':
				# go up
				# y_curr += 1
				x_curr -= 1
			elif action == 'S':
				# go south
				# y_curr -= 1
				x_curr += 1
			elif action == 'E':
				# go left
				# x_curr -= 1
				y_curr -= 1
			elif action == 'W':
				# go right
				# x_curr += 1
				y_curr += 1

		# if x_curr < 0 or x_curr >= self.w or y_curr < 0 or y_curr >= self.h:
		if x_curr < 0 or x_curr >= self.h or y_curr < 0 or y_curr >= self.w:
			x_curr = x_prev
			y_curr = y_prev
			reward = -1.
		else:
			# in this version of grid world 
			# the reward depends on the prev state
			reward = self.reward_grid[x_prev, y_prev]

		# state[x_curr, y_curr] = 1
			
		if self.step_count >= self.MAX_STEPS_PER_EPISODE:
			done = True
		else:
			done = False

		# self.state = state
		self.position = (x_curr, y_curr)
		self.done = done

		# return self.position, reward, done, {}
		obs = self.all_state[x_curr*self.w + y_curr]
		# return np.array(self.state), reward, done, {}
		return obs, reward, done, {}

	def close(self):
		# use this when you have a rendering viewer
		pass

	def render(self, mode='human'):
		# View rendering
		pass