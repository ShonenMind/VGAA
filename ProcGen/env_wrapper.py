import gym
import numpy as np
import procgen
import os
from datetime import datetime
from reward_loader import load_reward_fn

ENV_WRAPPER_LOG = "logs/env_wrapper_debug.log"

def log_to_file(message, log_file=ENV_WRAPPER_LOG):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

class ProcgenCoinRunEnvWrapper(gym.Env):
    def __init__(self, reward_code=None, num_levels=200, start_level=0, use_sequential_levels=False):
        # Use paint_vel_info=True
        self.env = gym.make('procgen:procgen-coinrun-v0',
                            num_levels=num_levels,
                            start_level=start_level,
                            use_sequential_levels=use_sequential_levels,
                            paint_vel_info=True)
        print("[DEBUG] Created environment with paint_vel_info=True")
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_code = reward_code
        self.reward_fn = self._load_reward_fn(reward_code) if reward_code else None
        
        # For pixel-based velocity tracking
        self.prev_player_pos = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Reset velocity tracking
        self.prev_player_pos = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Prepare state for reward function
        state = {"obs": obs}
        if "progress" in info:
            state["progress"] = info["progress"]

        # Calculate custom reward
        if self.reward_fn is not None:
            try:
                custom_reward = self.reward_fn(state, action, info, reward)
            except Exception as e:
                print("Reward function error:", e)
                log_to_file(f"WARNING: Reward function crashed: {str(e)}. Using default reward.")
                custom_reward = reward
        else:
            custom_reward = reward

        print(f"[DEBUG] Original reward: {reward}, Custom reward: {custom_reward}")
        log_to_file(f"Original reward: {reward}")

        return obs, float(custom_reward), done, info
    
    '''def _calculate_pixel_based_velocity(self, obs):
        """Fallback pixel-based velocity calculation"""
        try:
            from adaptive_pixel_utils import get_player_x_position, get_player_y_position
            
            current_x = get_player_x_position(obs)
            current_y = get_player_y_position(obs)
            current_pos = (current_x, current_y)
            
            if self.prev_player_pos is None:
                velocity = (0.0, 0.0)
            else:
                vel_x = current_pos[0] - self.prev_player_pos[0]
                vel_y = current_pos[1] - self.prev_player_pos[1]
                velocity = (vel_x, vel_y)
            
            self.prev_player_pos = current_pos
            return velocity
            
        except Exception as e:
            print(f"[WARNING] Pixel-based velocity calculation failed: {e}")
            return (0.0, 0.0)'''

    def render(self, mode='human'):
        frames = self.env.render(mode)
        if mode == 'rgb_array':
            if isinstance(frames, (list, tuple, np.ndarray)):
                return frames[0]
            return frames
        else:
            return frames

    def close(self):
        self.env.close()

    def _load_reward_fn(self, reward_code):
        print(f"[DEBUG] Attempting to load reward function...")
        reward_fn = load_reward_fn(reward_code)
        if reward_fn is None:
            print(f"[ERROR] Failed to load reward function. Code:\n{reward_code[:300]}...")
        else:
            print(f"[DEBUG] Successfully loaded reward_fn: {reward_fn}")
        return reward_fn