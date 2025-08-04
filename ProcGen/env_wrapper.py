import gym
import numpy as np
import procgen
#testing from vs code app (aanya)
from reward_loader import load_reward_fn

class ProcgenCoinRunEnvWrapper(gym.Env):
    def __init__(self, reward_code=None, num_levels=200, start_level=0, use_sequential_levels=False):
        self.env = gym.make('procgen:procgen-coinrun-v0',
                            num_levels=num_levels,
                            start_level=start_level,
                            use_sequential_levels=use_sequential_levels)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_code = reward_code
        self.reward_fn = self._load_reward_fn(reward_code) if reward_code else None

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        state = {"obs": obs}
        if "progress" in info:
            state["progress"] = info["progress"]

        try:
            custom_reward = self.reward_fn(state, action, info)
        except Exception as e:
            print("Reward function error:", e)
            custom_reward = 0.0

        return obs, np.array([custom_reward], dtype=np.float32), done, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def _load_reward_fn(self, reward_code):
        print("[DEBUG] Using shared load_reward_fn from reward_loader.py")
        return load_reward_fn(reward_code)
