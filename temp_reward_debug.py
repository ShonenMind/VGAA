import random
import math

def reward_fn(state, action, info):
    score = info.get("coins_collected", 0) * 5
    exploration_bonus = random.uniform(0.1, 0.5) if info.get("new_area_visited", False) else 0
    penalty = -1 * math.sqrt(abs(state.get("x", 0)))
    
    return score + exploration_bonus + penalty