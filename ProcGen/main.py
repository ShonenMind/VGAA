import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from env_wrapper import ProcgenCoinRunEnvWrapper
from rewards import get_random_reward_fn, get_refined_reward_fn




def load_reward_fn(reward_code_str):
   local_vars = {}
   exec(reward_code_str, {}, local_vars)
   return local_vars.get('reward_fn')




def make_train_env(reward_code):
   return ProcgenCoinRunEnvWrapper(
       reward_code=reward_code,
       num_levels=2,
       start_level=0,
       use_sequential_levels=True
   )




def make_eval_env():
   return ProcgenCoinRunEnvWrapper(
       num_levels=10,
       start_level=2,
       use_sequential_levels=True
   )




def summarize_trajectories(successful, unsuccessful):
   summary = (
       f"successful trajectories: {len(successful)}\n"
       f"unsuccessful trajectories: {len(unsuccessful)}\n"
       f"average length success: {np.mean([len(t) for t in successful]):.2f}\n"
       f"average length fail: {np.mean([len(t) for t in unsuccessful]):.2f}\n"
   )
   return summary




def collect_trajectories(env, model, n_episodes=10, success_threshold=5.0):
   successful = []
   unsuccessful = []


   for _ in range(n_episodes):
       obs = env.reset()
       traj = []
       total_reward = 0.0
       done = False


       while not done:
           action, _ = model.predict(obs, deterministic=True)
           next_obs, reward, done, info = env.step(action)


           state_dict = {"obs": obs.copy()}
           traj.append((state_dict, action))
           obs = next_obs
           total_reward += reward[0]


       if total_reward >= success_threshold:
           successful.append(traj)
       else:
           unsuccessful.append(traj)


   return successful, unsuccessful




def main():
   print("generating initial random reward function...")
   reward_code = get_random_reward_fn()
   print("initial reward function code:\n", reward_code)


   n_training_rounds = 3


   for round_idx in range(n_training_rounds):
       print(f"\n=== training round {round_idx + 1} ===")


       train_env = DummyVecEnv([lambda: make_train_env(reward_code)])
       train_env = VecMonitor(train_env)


       model = PPO("CnnPolicy", train_env, verbose=1, n_steps=2048)


       print("training ppo agent...")
       model.learn(total_timesteps=10000)
       model.save(f"ppo_procgen_model_round_{round_idx + 1}")


       eval_env = DummyVecEnv([make_eval_env])
       eval_env = VecMonitor(eval_env)


       print("evaluating ppo agent...")
       mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
       print(f"evaluation mean reward: {mean_reward:.2f} ± {std_reward:.2f}")


       print("collecting trajectories for tpe...")
       successful, unsuccessful = collect_trajectories(train_env, model, n_episodes=10)


       if len(successful) == 0 or len(unsuccessful) == 0:
           print("warning: not enough successful or unsuccessful trajectories for refinement. stopping.")
           break


       trajectory_summary = summarize_trajectories(successful, unsuccessful)
       print("trajectory summary:\n", trajectory_summary)


       print("generating refined reward function from llm...")
       refined_code = get_refined_reward_fn(reward_code, trajectory_summary)
       print("refined reward function code:\n", refined_code)


       reward_code = refined_code  # update reward function for next round


   print("\ntraining and refinement complete.")




if __name__ == "__main__":
   main()