"""
Gridworld Environment
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from point_dynamics import PointDynamics

class MultiGoal(gym.Env):
	"""
	Description:
		MultiGoal environment
	"""
	
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	# MAX_STEPS_PER_EPISODE = 20

	def __init__(self,
                 goal_reward=10,
                 actuation_cost_coeff=30.0,
                 distance_cost_coeff=1.0,
                 init_sigma=0.1):

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.goal_positions = np.array(
            (
                (5, 0),
                (-5, 0),
                (0, 5),
                (0, -5)
            ),
            dtype=np.float32)

        self.goal_threshold = 1.0
        self.goal_reward = goal_reward
        
		self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        self.reset()
        self.observation = None

        self._ax = None
        self._env_lines = []
        self.fixed_plots = None
        self.dynamic_plots = []
		self.seed(seed=1234)

	@property
    def observation_space(self):
        return spaces.Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            dtype=np.float32,
            shape=None)

	@property
    def action_space(self):
        return spaces.Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim, ),
            dtype=np.float32)

	def reset(self):
        
		unclipped_observation = (
            self.init_mu
            + self.init_sigma
            * np.random.normal(size=self.dynamics.s_dim))
        
		self.observation = np.clip(
            unclipped_observation,
            self.observation_space.low,
            self.observation_space.high)

        return self.observation

	def _init_plot(self):
        fig_env = plt.figure(figsize=(7, 7))
        self._ax = fig_env.add_subplot(111)
        self._ax.axis('equal')

        self._env_lines = []
        self._ax.set_xlim((-7, 7))
        self._ax.set_ylim((-7, 7))

        self._ax.set_title('Multigoal Environment')
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')

        self._plot_position_cost(self._ax)
		
	def step(self, action):
        action = action.ravel()

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high).ravel()

        observation = self.dynamics.forward(self.observation, action)
        observation = np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high)

        reward = self.compute_reward(observation, action)
        dist_to_goal = np.amin([
            np.linalg.norm(observation - goal_position)
            for goal_position in self.goal_positions
        ])
        done = dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward

        self.observation = np.copy(observation)

        return observation, reward, done, {'pos': observation}

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def close(self):
		# use this when you have a rendering viewer
		pass

	def render(self, mode='human'):
		# View rendering
		pass

	def render_rollouts(self, paths=()):
        """Render for rendering the past rollouts of the environment."""
        if self._ax is None:
            self._init_plot()

        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = []

        for path in paths:
            positions = np.stack(path['infos']['pos'])
            xx = positions[:, 0]
            yy = positions[:, 1]
            self._env_lines += self._ax.plot(xx, yy, 'b')

        plt.draw()
        plt.pause(0.01)
