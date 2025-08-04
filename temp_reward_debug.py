import random

def reward_fn(state, action, info):
    base_reward = info.get('coins_collected', 0) * 2
    randomness = random.uniform(0.5, 1.5)
    penalty = info.get('distance_to_next', 0) ** 0.5
    return base_reward * randomness - penalty