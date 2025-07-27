import numpy as np

def good_reward_func(state, action):
    obs = state["obs"]
    x_pos = np.mean(obs)  # placeholder!
    action_penalty = -np.linalg.norm(action) * 0.01
    return x_pos * 0.1 + action_penalty

def bad_reward_func(state, action):
    return 1.0
