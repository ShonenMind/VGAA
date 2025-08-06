import os
import numpy as np
import traceback
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
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

def record_agent_video(env, model, path="logs/ppo_agent.mp4", max_frames=1000):
    frames = []
    obs = env.reset()
    
    print(f"[DEBUG] Starting video recording to {path}")
    
    for i in range(max_frames):
        try:
            frame = env.render(mode="rgb_array")
            
            # Debug the frame shape and type
            if i == 0:  # Only print debug info for first frame to avoid spam
                print(f"[DEBUG] First frame type: {type(frame)}")
                if hasattr(frame, 'shape'):
                    print(f"[DEBUG] First frame shape: {frame.shape}")
                else:
                    print(f"[DEBUG] Frame has no shape attribute")
            
            # Check if frame is valid
            if frame is None:
                print(f"[WARNING] Frame {i} is None, skipping video recording")
                return
            
            if not hasattr(frame, 'shape'):
                print(f"[WARNING] Frame {i} has no shape attribute, skipping video recording")
                return
                
            if len(frame.shape) < 2:
                print(f"[WARNING] Frame {i} has insufficient dimensions ({frame.shape}), skipping video recording")
                return
                
            frames.append(frame)
            
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done[0]:
                obs = env.reset()
                
        except Exception as e:
            print(f"[ERROR] Exception during frame {i} capture: {e}")
            print("[WARNING] Skipping video recording due to rendering issues")
            return
    
    if not frames:
        print("[WARNING] No valid frames captured, skipping video save")
        return
        
    try:
        imageio.mimsave(path, frames, fps=30)
        print(f"[Video Saved] {path} ({len(frames)} frames)")
    except Exception as e:
        print(f"[ERROR] Failed to save video: {e}")
        print("[WARNING] Continuing without video")

def make_train_env(reward_code):
    return ProcgenCoinRunEnvWrapper(
        reward_code=reward_code,
        num_levels=100,
        start_level=0,
        use_sequential_levels=False
    )

def make_eval_env(reward_code=None):
    return ProcgenCoinRunEnvWrapper(
        reward_code=reward_code,
        num_levels=10,
        start_level=2,
        use_sequential_levels=False
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
    print(f"[DEBUG] Attempting to load reward function at {timestamp}")
    
    with open(REWARD_DEBUG_PATH, "a") as log:
        log.write(f"\n===== Reward Load Attempt [{timestamp}]")
        if round_idx is not None:
            log.write(f" | Round {round_idx + 1}")
        log.write(" =====\n")

        log.write("[DEBUG] reward_code_str just before exec:\n")
        log.write(reward_code + "\n")

        try:
            print("[DEBUG] Calling load_reward_fn()...")
            reward_fn = load_reward_fn(reward_code)
            print("[DEBUG] load_reward_fn() completed successfully")
        except Exception as e:
            error_msg = f"[ERROR] Exception while loading reward function:\n{traceback.format_exc()}\n"
            log.write(error_msg)
            print(error_msg)
            print("[FATAL] Reward function crash. Check reward_debug.txt.")
            return None

        if reward_fn is None:
            error_msg = "[FATAL] load_reward_fn() returned None.\nAvailable functions in global/local scope not found.\n"
            log.write(error_msg)
            print(error_msg)

        return reward_fn

def write_and_print_log(log_file, message, also_print=True):
    """Helper function to write to log file and optionally print to console"""
    log_file.write(message)
    log_file.flush()  # Ensure immediate write to disk
    if also_print:
        print(f"[LOG] {message.strip()}")

def main():
    print("[DEBUG] Starting main() function...")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    print("[DEBUG] Created logs directory")
    
    # Define log file paths - changed .log to .txt for consistency
    tpe_log_path = "logs/tpe_log.txt"
    edits_log_path = "logs/edits_log.txt"
    rounds_log_path = "logs/training_rounds.txt"
    
    print(f"[DEBUG] Log files will be created at:")
    print(f"  - TPE log: {tpe_log_path}")
    print(f"  - Edits log: {edits_log_path}")
    print(f"  - Rounds log: {rounds_log_path}")

    # Initialize log files with headers
    try:
        with open(tpe_log_path, "w") as f:
            f.write("TPE Log - Training Session\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write("="*50 + "\n")
        print(f"[DEBUG] Successfully initialized {tpe_log_path}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize TPE log: {e}")
        return

    try:
        with open(edits_log_path, "w") as f:
            f.write("Edits Log - Training Session\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write("="*50 + "\n")
        print(f"[DEBUG] Successfully initialized {edits_log_path}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize edits log: {e}")
        return

    try:
        with open(rounds_log_path, "w") as f:
            f.write("Training Rounds Log - Training Session\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write("="*50 + "\n")
        print(f"[DEBUG] Successfully initialized {rounds_log_path}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize rounds log: {e}")
        return

    # Generate initial reward function
    print("[DEBUG] Generating initial random reward function...")
    try:
        reward_code = get_random_reward_fn()
        print("[DEBUG] Successfully generated initial reward function")
        print("Initial reward function:\n", reward_code)
    except Exception as e:
        error_msg = f"[ERROR] Failed to generate initial reward function: {e}"
        print(error_msg)
        with open(rounds_log_path, "a") as f:
            write_and_print_log(f, f"FATAL ERROR: {error_msg}\n")
        return

    # Test loading initial reward function
    print("[DEBUG] Testing initial reward function...")
    reward_fn = try_load_reward_fn(reward_code)
    if reward_fn is None:
        error_msg = "[FATAL] Failed to load initial reward function."
        print(error_msg)
        with open(rounds_log_path, "a") as f:
            write_and_print_log(f, f"{error_msg}\n")
        return

    print("[DEBUG] Initial reward function loaded successfully")

    # Set number of training rounds
    n_training_rounds = 3
    print(f"[DEBUG] Starting {n_training_rounds} training rounds...")

    # Open log files for the training loop
    try:
        with open(tpe_log_path, "a") as tpe_log, \
             open(edits_log_path, "a") as edits_log, \
             open(rounds_log_path, "a") as rounds_log:
            
            print("[DEBUG] Opened all log files for writing")

            for round_idx in range(n_training_rounds):
                round_start_msg = f"\n=== Training Round {round_idx + 1} ===\n"
                print(round_start_msg.strip())
                write_and_print_log(rounds_log, round_start_msg, also_print=False)

                try:
                    # Create training environment
                    print(f"[DEBUG] Round {round_idx + 1}: Creating training environment...")
                    train_env = DummyVecEnv([lambda: make_train_env(reward_code) for _ in range(4)])
                    train_env = VecMonitor(train_env)
                    print(f"[DEBUG] Round {round_idx + 1}: Training environment created")

                    # Train model
                    print(f"[DEBUG] Round {round_idx + 1}: Creating and training PPO model...")
                    model = PPO(
                        "CnnPolicy",
                        train_env,
                        verbose=1,
                        n_steps=256,
                        batch_size=1024,
                        n_epochs=4,
                        learning_rate=5e-4,
                        ent_coef=0.01,
                    )

                    model.learn(total_timesteps=100000)
                    
                    model_path = f"ppo_model_round_{round_idx + 1}"
                    model.save(model_path)
                    print(f"[DEBUG] Round {round_idx + 1}: Model saved to {model_path}")

                    # Evaluate model
                    print(f"[DEBUG] Round {round_idx + 1}: Creating evaluation environment...")
                    eval_env = DummyVecEnv([lambda: make_eval_env(reward_code)])
                    eval_env = VecMonitor(eval_env)
                    
                    print(f"[DEBUG] Round {round_idx + 1}: Evaluating policy...")
                    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
                    eval_msg = f"Round {round_idx + 1}: Eval Reward: {mean_reward:.2f} ± {std_reward:.2f}"
                    print(eval_msg)
                    write_and_print_log(rounds_log, f"{eval_msg}\n", also_print=False)

                    # Record video (non-critical - don't let this stop training)
                    print(f"[DEBUG] Round {round_idx + 1}: Recording agent video...")
                    video_path = f"logs/ppo_round_{round_idx + 1}.mp4"
                    try:
                        record_agent_video(eval_env, model, path=video_path, max_frames=500)
                    except Exception as e:
                        video_error_msg = f"Round {round_idx + 1}: Video recording failed: {str(e)}"
                        print(f"[WARNING] {video_error_msg}")
                        write_and_print_log(rounds_log, f"WARNING: {video_error_msg}\n", also_print=False)
                        print("[INFO] Continuing training without video...")

                    # Collect trajectories
                    print(f"[DEBUG] Round {round_idx + 1}: Collecting trajectories...")
                    successful, unsuccessful = collect_trajectories(train_env, model, n_episodes=10, round_idx=round_idx)
                    
                    traj_msg = f"Round {round_idx + 1}: Collected {len(successful)} successful, {len(unsuccessful)} unsuccessful trajectories"
                    print(traj_msg)
                    write_and_print_log(rounds_log, f"{traj_msg}\n", also_print=False)

                    if not successful or not unsuccessful:
                        insufficient_msg = f"Round {round_idx + 1}: Insufficient successful/unsuccessful trajectories for TPE. Stopping."
                        print(insufficient_msg)
                        write_and_print_log(rounds_log, f"{insufficient_msg}\n")
                        write_and_print_log(tpe_log, f"Round {round_idx + 1}: STOPPED - Insufficient trajectories\n")
                        write_and_print_log(edits_log, f"Round {round_idx + 1}: STOPPED - Insufficient trajectories\n")
                        break

                    # Reload reward function for TPE
                    print(f"[DEBUG] Round {round_idx + 1}: Reloading reward function for TPE...")
                    reward_fn = try_load_reward_fn(reward_code, round_idx=round_idx)
                    if reward_fn is None:
                        reload_error_msg = f"Round {round_idx + 1}: Failed to reload reward function."
                        print(f"[FATAL] {reload_error_msg}")
                        write_and_print_log(rounds_log, f"FATAL: {reload_error_msg}\n")
                        write_and_print_log(tpe_log, f"Round {round_idx + 1}: FATAL - Reward function reload failed\n")
                        return

                    # Perform TPE
                    print(f"[DEBUG] Round {round_idx + 1}: Running trajectory preference evaluation...")
                    passed, accuracy = trajectory_preference_evaluation(reward_fn, successful, unsuccessful)
                    
                    tpe_msg = f"Round {round_idx + 1}: TPE Score = {accuracy:.2f}, Passed = {passed}"
                    print(tpe_msg)
                    write_and_print_log(tpe_log, f"{tpe_msg}\n")
                    write_and_print_log(rounds_log, f"{tpe_msg}\n", also_print=False)

                    # Trajectory summary
                    summary = summarize_trajectories(successful, unsuccessful)
                    print(f"[DEBUG] Round {round_idx + 1}: Trajectory Summary:")
                    print(summary)
                    write_and_print_log(rounds_log, f"Trajectory Summary:\n{summary}\n", also_print=False)

                    # Decision making for reward function revision
                    if not passed:
                        decision_msg = f"Round {round_idx + 1}: TPE < 0.8 → Reactive correction required."
                        print(decision_msg)
                        write_and_print_log(rounds_log, f"{decision_msg}\n", also_print=False)
                        
                        print(f"[DEBUG] Round {round_idx + 1}: Generating reactive reward function...")
                        refined_code = get_reactive_reward_fn(reward_code, summary)
                        
                        edit_msg = f"Round {round_idx + 1}: TPE={accuracy:.2f}, Edit=Reactive"
                        write_and_print_log(edits_log, f"{edit_msg}\n")
                    else:
                        if should_proactively_revise(reward_code):
                            decision_msg = f"Round {round_idx + 1}: TPE passed but choosing to refine proactively."
                            print(decision_msg)
                            write_and_print_log(rounds_log, f"{decision_msg}\n", also_print=False)
                            
                            print(f"[DEBUG] Round {round_idx + 1}: Generating proactive reward function...")
                            refined_code = get_proactive_reward_fn(reward_code)
                            
                            edit_msg = f"Round {round_idx + 1}: TPE={accuracy:.2f}, Edit=Proactive"
                            write_and_print_log(edits_log, f"{edit_msg}\n")
                        else:
                            decision_msg = f"Round {round_idx + 1}: TPE passed. Keeping current reward function."
                            print(decision_msg)
                            write_and_print_log(rounds_log, f"{decision_msg}\n", also_print=False)
                            
                            edit_msg = f"Round {round_idx + 1}: TPE={accuracy:.2f}, Edit=None"
                            write_and_print_log(edits_log, f"{edit_msg}\n")
                            refined_code = reward_code

                    reward_code = refined_code
                    print(f"[DEBUG] Round {round_idx + 1}: Updated reward code for next round")

                except Exception as e:
                    error_msg = f"Round {round_idx + 1}: EXCEPTION OCCURRED: {str(e)}\n{traceback.format_exc()}"
                    print(f"[ERROR] {error_msg}")
                    write_and_print_log(rounds_log, f"ERROR: {error_msg}\n")
                    write_and_print_log(tpe_log, f"Round {round_idx + 1}: ERROR - {str(e)}\n")
                    write_and_print_log(edits_log, f"Round {round_idx + 1}: ERROR - {str(e)}\n")
                    # Continue to next round instead of stopping completely
                    continue

            # Final completion message
            completion_msg = f"\nTraining and refinement complete at {datetime.now().isoformat()}\n"
            print(completion_msg.strip())
            write_and_print_log(rounds_log, completion_msg, also_print=False)
            write_and_print_log(tpe_log, completion_msg, also_print=False)
            write_and_print_log(edits_log, completion_msg, also_print=False)

    except Exception as e:
        fatal_error_msg = f"[FATAL ERROR] Failed to open log files: {e}"
        print(fatal_error_msg)
        return

    print("[DEBUG] main() function completed successfully")

if __name__ == "__main__":
    main()