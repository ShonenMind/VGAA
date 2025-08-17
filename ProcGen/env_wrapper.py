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
        # Use paint_vel_info=True (even though we'll use pixel-based tracking for now)
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
        
        # Extract velocity from painted velocity info
        velocity = self._extract_velocity_from_painted_info(obs)
        
        # Add velocity information to info
        info['velocity_x'] = velocity[0]
        info['velocity_y'] = velocity[1]
        info['velocity_magnitude'] = (velocity[0]**2 + velocity[1]**2)**0.5
        info['velocity'] = velocity

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

        print(f"[DEBUG] Original reward: {reward}, Custom reward: {custom_reward}, Velocity: {velocity}")
        log_to_file(f"Original reward: {reward}, Custom reward: {custom_reward}, Velocity: {velocity}")

        return obs, float(custom_reward), done, info

    def _extract_velocity_from_painted_info(self, obs):
        """Extract velocity from painted velocity info using the official encoding formula"""
        try:
            # Extract the painted velocity area (top-left corner)
            painted_area = obs[:20, :30, :]  # Adjust size as needed
            
            # Find pixels that are in the gray velocity range (around 127)
            gray_mask = (painted_area[:,:,0] >= 100) & (painted_area[:,:,0] <= 154) & \
                       (painted_area[:,:,1] >= 100) & (painted_area[:,:,1] <= 154) & \
                       (painted_area[:,:,2] >= 100) & (painted_area[:,:,2] <= 154)
            
            if not np.any(gray_mask):
                # Fall back to pixel-based tracking if no painted velocity found
                return self._calculate_pixel_based_velocity(obs)
            
            # Get the painted velocity pixels
            gray_pixels = painted_area[gray_mask]
            
            if len(gray_pixels) == 0:
                return self._calculate_pixel_based_velocity(obs)
            
            # Use the most common gray values (in case there are multiple)
            unique_colors, counts = np.unique(gray_pixels.reshape(-1, 3), axis=0, return_counts=True)
            dominant_color = unique_colors[np.argmax(counts)]
            
            # Decode velocity using the official formula:
            # velocity = (color - 127) / 128 * max_velocity
            
            # Estimate max_velocity (typical values for CoinRun are around 10-20)
            # You may need to adjust this based on observation
            max_velocity = 15.0
            
            # Extract velocity components from R and G channels
            # (assuming x velocity in R channel, y velocity in G channel)
            vel_x = ((dominant_color[0] - 127) / 128.0) * max_velocity
            vel_y = ((dominant_color[1] - 127) / 128.0) * max_velocity
            
            # Clamp velocities to reasonable ranges
            vel_x = np.clip(vel_x, -max_velocity, max_velocity)
            vel_y = np.clip(vel_y, -max_velocity, max_velocity)
            
            return (float(vel_x), float(vel_y))
            
        except Exception as e:
            print(f"[WARNING] Painted velocity extraction failed: {e}")
            # Fall back to pixel-based tracking
            return self._calculate_pixel_based_velocity(obs)
    
    def _calculate_pixel_based_velocity(self, obs):
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
            return (0.0, 0.0)

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