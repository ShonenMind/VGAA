import gym
import numpy as np
import procgen
#sorry im testing if i can commit

class ProcgenCoinRunEnvWrapper(gym.Env):
   def __init__(self, reward_code=None, num_levels=200, start_level=0, use_sequential_levels=True):
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
       obs, base_reward, done, info = self.env.step(action)
       if self.reward_fn:
           custom_reward = self.reward_fn(obs, action, info)
           reward = custom_reward
       else:
           reward = base_reward
       return obs, reward, done, info


   def render(self, mode='human'):
       return self.env.render(mode)


   def close(self):
       self.env.close()
  
   def _load_reward_fn(self, reward_code):
       local_vars = {}
       exec(reward_code, {}, local_vars)
       return local_vars.get('reward_fn')


