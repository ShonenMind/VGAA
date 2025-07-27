import gym


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy


#this trains on 2 levels of the procgen coinrun environment and evaluates on 10 other levels
def make_train_env():
   return gym.make('procgen:procgen-coinrun-v0',
                   num_levels=2,         # train on 2 levels: 0 and 1
                   start_level=0,
                   use_sequential_levels=True)


def make_eval_env():
   return gym.make('procgen:procgen-coinrun-v0',
                   num_levels=10,        # evaluate on 10 levels: 2 through 11
                   start_level=2,
                   use_sequential_levels=True)


def main():
   # training environment
   train_env = DummyVecEnv([make_train_env])
   train_env = VecMonitor(train_env)


   # create ppo model
   model = PPO("CnnPolicy", train_env, verbose=2, n_steps=2048)


   # train
   print("training...")
   model.learn(total_timesteps=10000)

   #saving model
   model.save("ppo_procgen_model")

   # evaluation environment (separate env for evaluation)
   eval_env = DummyVecEnv([make_eval_env])
   eval_env = VecMonitor(eval_env)


   print("evaluating...")
   mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
   print(f"mean reward: {mean_reward:.2f} ± {std_reward:.2f}")


if __name__ == "__main__":
   main()
