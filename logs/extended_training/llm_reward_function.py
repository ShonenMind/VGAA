def reward_fn(state, action, info, original_reward=0):
    import numpy as np
    import random

    # Shape the original reward from the environment
    shaped_reward = np.clip(original_reward, -1, 1) 

    # Encourage exploration
    exploration_bonus = 0.01

    # Encourage action diversity
    action_diversity_bonus = 0.05 if action != state.get('prev_action', 0) else 0

    # Extract observation for velocity info
    obs = state.get('obs', None)

    velocity_bonus = 0
    # Extract velocity from painted velocity boxes (top-left corner)
    if obs is not None and obs.shape == (64, 64, 3):
        h_vel_pixel = obs[4, 4]  # Center of left box (horizontal velocity)
        v_vel_pixel = obs[4, 12]  # Center of right box (vertical velocity)
        
        # Normalize to -1 to +1
        h_velocity = (np.mean(h_vel_pixel) - 128) / 128.0
        v_velocity = (np.mean(v_vel_pixel) - 128) / 128.0
        
        # Reward rightward movement and jumping, penalize going backwards or falling
        velocity_bonus = max(0, h_velocity) * 0.2 + max(0, v_velocity) * 0.1

    # Reward for level completion
    level_completion_bonus = info.get('prev_level_complete', 0) * 2.0

    # Add some small randomness to the reward for exploration
    noise = random.uniform(-0.005, 0.005)

    # Update the previous action in the state
    state['prev_action'] = action

    return shaped_reward + exploration_bonus + action_diversity_bonus + velocity_bonus + level_completion_bonus + noise