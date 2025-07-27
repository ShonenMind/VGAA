import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rewards import good_reward_func, bad_reward_func


def make_eval_env():
    return gym.make('procgen:procgen-coinrun-v0',
                    num_levels=10,
                    start_level=2,
                    use_sequential_levels=True)


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
            next_obs, reward, done, _ = env.step(action)

            # Store just the raw obs for now
            state_dict = {"obs": obs.copy()}
            traj.append((state_dict, action))
            obs = next_obs
            total_reward += reward[0]

        if total_reward >= success_threshold:
            successful.append(traj)
        else:
            unsuccessful.append(traj)

    return successful, unsuccessful


def calculate_average_return(trajectory, reward_func, gamma=0.99):
    cumulative_return = 0.0
    for i, (state, action) in enumerate(trajectory):
        reward = reward_func(state, action)
        cumulative_return += (gamma ** i) * reward
    return cumulative_return / len(trajectory)


def trajectory_preference_evaluation(reward_func, successful, unsuccessful, threshold=0.8):
    print(f"\nEvaluating {reward_func.__name__}...")

    success_returns = [calculate_average_return(t, reward_func) for t in successful]
    failure_returns = [calculate_average_return(t, reward_func) for t in unsuccessful]

    comparisons = 0
    correct = 0
    for s_ret in success_returns:
        for f_ret in failure_returns:
            comparisons += 1
            if s_ret > f_ret:
                correct += 1

    if comparisons == 0:
        return True, 1.0

    accuracy = correct / comparisons
    passed = accuracy >= threshold

    print(f"Accuracy: {accuracy:.2f} → {'PASS' if passed else 'FAIL'}")
    return passed, accuracy


def main():
    print("Loading model...")
    model = PPO.load("ppo_procgen_model")

    print("Creating environment...")
    env = DummyVecEnv([make_eval_env])

    print("Collecting trajectories...")
    succ, fail = collect_trajectories(env, model, n_episodes=10)

    if len(succ) == 0 or len(fail) == 0:
        print("Not enough successful or failed trajectories to evaluate") #debug
        return

    # test different reward functions
    trajectory_preference_evaluation(good_reward_func, succ, fail)
    trajectory_preference_evaluation(bad_reward_func, succ, fail)


if __name__ == "__main__":
    main()