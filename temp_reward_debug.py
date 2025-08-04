import math
import random

def reward_fn(state, action, info):
    coin_count = info.get('coins_collected', 0)
    current_velocity = state.get('velocity', 0)
    rng_bonus = random.uniform(0.1, 1.0)
    reward = coin_count * math.sin(current_velocity) * rng_bonus
    return reward