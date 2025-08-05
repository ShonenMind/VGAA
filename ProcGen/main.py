import os
import numpy as np
import traceback
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from env_wrapper import ProcgenCoinRunEnvWrapper
from rewards import (
    get_random_reward_fn,
    get_reactive_reward_fn,
    get_proactive_reward_fn,
    should_proactively_revise
)
from tpe import (
    load_reward_fn,
    collect_trajectories,
    trajectory_preference_evaluation
)

REWARD_DEBUG_PATH = "logs/reward_debug.txt"

def make_train_env(reward_code):
    return ProcgenCoinRunEnvWrapper(
        reward_code=reward_code,
        num_levels=5,
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
    return (
        f"successful trajectories: {len(successful)}\n"
        f"unsuccessful trajectories: {len(unsuccessful)}\n"
        f"average length success: {np.mean([len(t) for t in successful]):.2f}\n"
        f"average length fail: {np.mean([len(t) for t in unsuccessful]):.2f}\n"
    )

def try_load_reward_fn(reward_code, round_idx=None):
    timestamp = datetime.now().isoformat()
    with open(REWARD_DEBUG_PATH, "a") as log:
        log.write(f"\n===== Reward Load Attempt [{timestamp}]")
        if round_idx is not None:
            log.write(f" | Round {round_idx + 1}")
        log.write(" =====\n")

        log.write("[DEBUG] reward_code_str just before exec:\n")
        log.write(reward_code + "\n")

        try:
            reward_fn = load_reward_fn(reward_code)
        except Exception as e:
            log.write("[ERROR] Exception while loading reward function:\n")
            log.write(traceback.format_exc() + "\n")
            print("[FATAL] Reward function crash. Check reward_debug.txt.")
            return None

        if reward_fn is None:
            log.write("[FATAL] load_reward_fn() returned None.\n")
            log.write("Available functions in global/local scope not found.\n")

        return reward_fn

def main():
    os.makedirs("logs", exist_ok=True)
    tpe_log_path = "logs/tpe_log.txt"
    edits_log_path = "logs/edits_log.txt"
    rounds_log_path = "logs/training_rounds.log"   # New log for round decisions

    with open(tpe_log_path, "w") as tpe_log, open(edits_log_path, "w") as edits_log, open(rounds_log_path, "w") as rounds_log:
        print("Generating initial random reward function...")
        reward_code = get_random_reward_fn()
        print("Initial reward function:\n", reward_code)

        reward_fn = try_load_reward_fn(reward_code)
        if reward_fn is None:
            print("[FATAL] Failed to load initial reward function.")
            return

        n_training_rounds = 7

        for round_idx in range(n_training_rounds):
            print(f"\n=== Training Round {round_idx + 1} ===")

            train_env = DummyVecEnv([lambda: make_train_env(reward_code)])
            train_env = VecMonitor(train_env)

            model = PPO("CnnPolicy", train_env, verbose=1, n_steps=2048)
            model.learn(total_timesteps=10000)
            model.save(f"ppo_model_round_{round_idx + 1}")

            eval_env = DummyVecEnv([make_eval_env])
            eval_env = VecMonitor(eval_env)
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
            print(f"Eval Reward: {mean_reward:.2f} ± {std_reward:.2f}")

            successful, unsuccessful = collect_trajectories(train_env, model, n_episodes=10)
            if not successful or not unsuccessful:
                print("Insufficient successful/unsuccessful trajectories for TPE. Stopping.")
                rounds_log.write(f"Round {round_idx + 1}: Stopped early due to insufficient trajectories.\n")
                break

            reward_fn = try_load_reward_fn(reward_code, round_idx=round_idx)
            if reward_fn is None:
                print("[FATAL] Reward function failed to reload after training.")
                rounds_log.write(f"Round {round_idx + 1}: Failed to reload reward function.\n")
                return

            passed, accuracy = trajectory_preference_evaluation(reward_fn, successful, unsuccessful)

            tpe_log.write(f"Round {round_idx + 1}: TPE Score = {accuracy:.2f}, Passed = {passed}\n")
            tpe_log.flush()

            summary = summarize_trajectories(successful, unsuccessful)
            print("\nTrajectory Summary:\n", summary)

            if not passed:
                print("TPE < 0.8 → Reactive correction required.")
                refined_code = get_reactive_reward_fn(reward_code, summary)
                edits_log.write(f"Round {round_idx + 1}: TPE={accuracy:.2f}, Edit=Reactive\n")
                rounds_log.write(f"Round {round_idx + 1}: TPE={accuracy:.2f}, Decision=Reactive correction\n")
            else:
                if should_proactively_revise(reward_code):
                    print("TPE passed but choosing to refine proactively.")
                    refined_code = get_proactive_reward_fn(reward_code)
                    edits_log.write(f"Round {round_idx + 1}: TPE={accuracy:.2f}, Edit=Proactive\n")
                    rounds_log.write(f"Round {round_idx + 1}: TPE={accuracy:.2f}, Decision=Proactive revision\n")
                else:
                    print("TPE passed. Keeping current reward function.")
                    edits_log.write(f"Round {round_idx + 1}: TPE={accuracy:.2f}, Edit=None\n")
                    rounds_log.write(f"Round {round_idx + 1}: TPE={accuracy:.2f}, Decision=Keep current\n")
                    refined_code = reward_code

            edits_log.flush()
            rounds_log.flush()
            reward_code = refined_code

        print("\nTraining and refinement complete.")

if __name__ == "__main__":
    main() 
