from gym.envs.registration import register
import numpy as np

# r(s,a) = 10 if s is (1,4) or (3,4) (This is sent as the reward_grid parameter)
# r(s,a) = -1 if s is a boundary state (This is coded in the step function itself)
# r(s,a) = 0 otherwise (This is sent as the reward_grid parameter)

# Specification for the GridWorld version
# in Sutton Example 3.5: Gridworld 
width = 5
height = 5

# Define the reward_grid s.t.
# reward_grid[x_start, y_start] = reward_value
reward_grid_v0 = np.zeros((width, height))
reward_grid_v0[(0,1)] = 10.0
reward_grid_v0[(0,3)] = 5.0

# Define jump transitions
jump_transitions_v0 = dict()
jump_transitions_v0[(0,1)] = (4,1)
jump_transitions_v0[(0,3)] = (2,3)

register(
    id='GridWorld-v0',
    entry_point='gridworld_env.gridworld:GridWorld',
    kwargs={
        'reward_grid': reward_grid_v0,
        'jump_transitions': jump_transitions_v0
    }
)