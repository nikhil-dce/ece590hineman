from gym.envs.registration import register
import numpy as np

# width = 5
# height = 5


# reward_grid_v0 = np.zeros((width, height))
# reward_grid_v0[(0,1)] = 10.0
# reward_grid_v0[(0,3)] = 5.0

# # Define jump transitions
# jump_transitions_v0 = dict()
# jump_transitions_v0[(0,1)] = (4,1)
# jump_transitions_v0[(0,3)] = (2,3)

register(
    id='MultiGoal-v0',
    entry_point='multigoal_env.multigoal:MultiGoal',
    kwargs={
        # 'reward_grid': reward_grid_v0,
        # 'jump_transitions': jump_transitions_v0
    }
)