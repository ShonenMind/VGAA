import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env_wrapper import ProcgenCoinRunEnvWrapper
from rewards import get_random_reward_fn, get_reactive_reward_fn, get_proactive_reward_fn, should_proactively_revise
from reward_loader import load_reward_fn
def collect_trajectories(env, model, n_episodes=10, round_idx=None):
    successful = []
    unsuccessful = []

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/tpe_log.txt"
    with open(log_path, "a") as log_file:
        log_file.write(f"\n[Collecting trajectories - Round {round_idx + 1 if round_idx is not None else '?'}]\n")

        for ep in range(n_episodes):
            obs = env.reset()
            traj = []
            done = [False]
            total_progress = 0.0

            while not np.all(done):
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, done, infos = env.step(action)

                info = infos[0]
                state_dict = {"obs": obs.copy()}

                if "progress" in info:
                    state_dict["progress"] = info["progress"]
                    total_progress = max(total_progress, info["progress"])

                traj.append((state_dict, action))
                obs = next_obs

            log_file.write(f"Episode {ep+1}: final progress = {total_progress:.3f}\n")

            if total_progress >= 0.2:
                successful.append(traj)
            else:
                unsuccessful.append(traj)

    return successful, unsuccessful

def calculate_average_return(trajectory, reward_func, gamma=0.99):
    cumulative_return = 0.0
    for i, (state, action) in enumerate(trajectory):
        try:
            reward = reward_func(state, action, {})
        except Exception as e:
            print("[ERROR] Reward function error at step", i)
            print("State:", state)
            print("Action:", action)
            print("Error:", e)
            reward = 0.0
        cumulative_return += (gamma ** i) * reward
    return cumulative_return / len(trajectory)

def trajectory_preference_evaluation(reward_func, successful, unsuccessful, threshold=0.8):
    print(f"\nEvaluating reward function: {getattr(reward_func, '__name__', 'LLM-generated')}")

    success_returns = [calculate_average_return(t, reward_func) for t in successful]
    failure_returns = [calculate_average_return(t, reward_func) for t in unsuccessful]

    comparisons = 0
    correct = 0
    for s_ret in success_returns:
        for f_ret in failure_returns:
            comparisons += 1
            if s_ret > f_ret:
                correct += 1

    accuracy = correct / comparisons if comparisons > 0 else 1.0
    passed = accuracy >= threshold

    print(f"Accuracy: {accuracy:.2f} → {'PASS' if passed else 'FAIL'}")
    return passed, accuracy

def main():
    print("Generating initial random LLM reward function...")
    reward_code = get_random_reward_fn()
    print("Generated reward function code:\n", reward_code)

    reward_fn = load_reward_fn(reward_code)
    if reward_fn is None:
        print("ERROR: Could not load reward function from generated code!")
        return

    print("Loading trained PPO model...")
    model = PPO.load("ppo_procgen_model")

    print("Creating Procgen environment with custom LLM reward...")
    env = DummyVecEnv([lambda: ProcgenCoinRunEnvWrapper(reward_code=reward_code)])

    print("Collecting trajectories with current reward function...")
    successful, unsuccessful = collect_trajectories(env, model, n_episodes=10)

    if len(successful) == 0 or len(unsuccessful) == 0:
        print("Not enough successful or unsuccessful trajectories to perform TPE evaluation.")
        return

    print("Evaluating trajectory preference with generated reward function...")
    passed, accuracy = trajectory_preference_evaluation(reward_fn, successful, unsuccessful)

    trajectory_summary = (
        f"Successful trajectories: {len(successful)}\n"
        f"Unsuccessful trajectories: {len(unsuccessful)}\n"
        f"Example success avg length: {np.mean([len(t) for t in successful]):.2f}\n"
        f"Example failure avg length: {np.mean([len(t) for t in unsuccessful]):.2f}\n"
    )
    print("\nTrajectory summary to be used for refinement:\n", trajectory_summary)

    if not passed:
        print("Reward function failed TPE // must fix")
        refined_reward_code = get_reactive_reward_fn(reward_code, trajectory_summary)
        print("Refined reward function code:\n", refined_reward_code)
    else:
        print("Reward function passed TPE (>=0.8) // optional proactive revision")
        if should_proactively_revise(reward_code):
            refined_reward_code = get_proactive_reward_fn(reward_code)
            print("Proactively revised reward function code:\n", refined_reward_code)
        else:
            print("Keeping current reward function for next iteration.")

if __name__ == "__main__":
    main()