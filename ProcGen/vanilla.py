import os
import numpy as np
import traceback
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import gym
import procgen

VANILLA_LOG_PATH = "logs/vanilla/"

def record_agent_video(env, model, path="logs/vanilla/ppo_agent.mp4", max_frames=1000):
    """Record agent gameplay video"""
    frames = []
    obs = env.reset()
    
    print(f"[DEBUG] Starting video recording to {path}")
    
    for i in range(max_frames):
        try:
            frame = env.render(mode="rgb_array")
            
            if i == 0:  # Only print debug info for first frame
                print(f"[DEBUG] First frame type: {type(frame)}")
                if hasattr(frame, 'shape'):
                    print(f"[DEBUG] First frame shape: {frame.shape}")
                else:
                    print(f"[DEBUG] Frame has no shape attribute")
            
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

class VanillaProcgenEnv(gym.Env):
    """Basic ProcGen CoinRun environment without custom rewards"""
    def __init__(self, num_levels=100, start_level=0, use_sequential_levels=False):
        self.env = gym.make('procgen:procgen-coinrun-v0',
                            num_levels=num_levels,
                            start_level=start_level,
                            use_sequential_levels=use_sequential_levels,
                            paint_vel_info=True)
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        print("[DEBUG] Created vanilla ProcGen environment (no custom rewards)")

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Use the original environment reward without modification
        return obs, float(reward), done, info

    def render(self, mode='human'):
        frames = self.env.render(mode)
        if mode == 'rgb_array':
            if isinstance(frames, (list, tuple, np.ndarray)):
                return frames[0] if len(frames) > 0 else frames
            return frames
        else:
            return frames

    def close(self):
        self.env.close()

def make_train_env():
    """Create training environment"""
    return VanillaProcgenEnv(
        num_levels=100,
        start_level=0,
        use_sequential_levels=False
    )

def make_eval_env():
    """Create evaluation environment"""
    return VanillaProcgenEnv(
        num_levels=10,
        start_level=2,
        use_sequential_levels=False
    )

def collect_trajectories(env, model, n_episodes=10, round_idx=None):
    """Collect successful and unsuccessful trajectories for analysis"""
    successful = []
    unsuccessful = []
    
    # Also collect reward data
    episode_rewards = []
    step_rewards = []

    log_path = os.path.join(VANILLA_LOG_PATH, "vanilla_trajectories.txt")
    rewards_log_path = os.path.join(VANILLA_LOG_PATH, "vanilla_rewards.txt")
    
    with open(log_path, "a") as log_file, open(rewards_log_path, "a") as rewards_file:
        log_file.write(f"\n[Collecting vanilla trajectories - Round {round_idx + 1 if round_idx is not None else '?'}]\n")
        rewards_file.write(f"\n[Vanilla Rewards - Round {round_idx + 1 if round_idx is not None else '?'}]\n")

        for ep in range(n_episodes):
            obs = env.reset()
            traj = []
            done = [False]
            total_progress = 0.0
            episode_reward = 0.0
            episode_step_rewards = []

            while not np.all(done):
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, done, infos = env.step(action)

                # Log the actual reward received
                actual_reward = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                episode_reward += actual_reward
                episode_step_rewards.append(actual_reward)
                step_rewards.append(actual_reward)

                info = infos[0]
                state_dict = {"obs": obs.copy()}

                if "progress" in info:
                    state_dict["progress"] = info["progress"]
                    total_progress = max(total_progress, info["progress"])

                # Store reward in trajectory for analysis
                traj.append((state_dict, action, actual_reward))
                obs = next_obs

            episode_rewards.append(episode_reward)
            
            # Log episode summary
            log_file.write(f"Episode {ep+1}: final progress = {total_progress:.3f}, total reward = {episode_reward:.3f}\n")
            rewards_file.write(f"Episode {ep+1}: total_reward={episode_reward:.3f}, steps={len(episode_step_rewards)}, avg_step_reward={np.mean(episode_step_rewards):.4f}\n")

            if total_progress >= 0.2:
                successful.append(traj)
            else:
                unsuccessful.append(traj)
        
        # Log summary statistics
        rewards_file.write(f"Round {round_idx + 1} Summary:\n")
        rewards_file.write(f"  Mean episode reward: {np.mean(episode_rewards):.3f}\n")
        rewards_file.write(f"  Std episode reward: {np.std(episode_rewards):.3f}\n")
        rewards_file.write(f"  Min episode reward: {np.min(episode_rewards):.3f}\n")
        rewards_file.write(f"  Max episode reward: {np.max(episode_rewards):.3f}\n")
        rewards_file.write(f"  Mean step reward: {np.mean(step_rewards):.4f}\n")
        rewards_file.write(f"  Total steps: {len(step_rewards)}\n")

    return successful, unsuccessful

def write_and_print_log(log_file, message, also_print=True):
    """Helper function to write to log file and optionally print to console"""
    log_file.write(message)
    log_file.flush()
    if also_print:
        print(f"[LOG] {message.strip()}")

def main():
    print("[DEBUG] Starting vanilla PPO training on ProcGen CoinRun...")
    
    # Create logs directory
    os.makedirs(VANILLA_LOG_PATH, exist_ok=True)
    print(f"[DEBUG] Created vanilla logs directory: {VANILLA_LOG_PATH}")
    
    # Define log file paths
    training_log_path = os.path.join(VANILLA_LOG_PATH, "training_rounds.txt")
    trajectories_log_path = os.path.join(VANILLA_LOG_PATH, "trajectories.txt")
    rewards_log_path = os.path.join(VANILLA_LOG_PATH, "vanilla_rewards.txt")
    
    print(f"[DEBUG] Log files will be created at:")
    print(f"  - Training log: {training_log_path}")
    print(f"  - Trajectories log: {trajectories_log_path}")
    print(f"  - Rewards log: {rewards_log_path}")

    # Initialize log files
    try:
        with open(training_log_path, "w") as f:
            f.write("Vanilla PPO Training Log\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write("="*50 + "\n")
        print(f"[DEBUG] Successfully initialized {training_log_path}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize training log: {e}")
        return

    try:
        with open(trajectories_log_path, "w") as f:
            f.write("Vanilla Trajectories Log\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write("="*50 + "\n")
        print(f"[DEBUG] Successfully initialized {trajectories_log_path}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize trajectories log: {e}")
        return

    try:
        with open(rewards_log_path, "w") as f:
            f.write("Vanilla Rewards Log\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write("="*50 + "\n")
        print(f"[DEBUG] Successfully initialized {rewards_log_path}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize rewards log: {e}")
        return

    # Set number of training rounds (same as LLM experiments)
    n_training_rounds = 3
    print(f"[DEBUG] Starting {n_training_rounds} vanilla training rounds...")

    # Open log files for the training loop
    try:
        with open(training_log_path, "a") as training_log:
            
            print("[DEBUG] Opened training log file for writing")

            for round_idx in range(n_training_rounds):
                round_start_msg = f"\n=== Vanilla Training Round {round_idx + 1} ===\n"
                print(round_start_msg.strip())
                write_and_print_log(training_log, round_start_msg, also_print=False)

                try:
                    # Create training environment
                    print(f"[DEBUG] Round {round_idx + 1}: Creating vanilla training environment...")
                    train_env = DummyVecEnv([lambda: make_train_env() for _ in range(4)])
                    train_env = VecMonitor(train_env)
                    print(f"[DEBUG] Round {round_idx + 1}: Training environment created")

                    # Train model (same hyperparameters as LLM experiments)
                    print(f"[DEBUG] Round {round_idx + 1}: Creating and training vanilla PPO model...")
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
                    
                    model_path = os.path.join(VANILLA_LOG_PATH, f"vanilla_ppo_model_round_{round_idx + 1}")
                    model.save(model_path)
                    print(f"[DEBUG] Round {round_idx + 1}: Vanilla model saved to {model_path}")

                    # Evaluate model
                    print(f"[DEBUG] Round {round_idx + 1}: Creating evaluation environment...")
                    eval_env = DummyVecEnv([lambda: make_eval_env()])
                    eval_env = VecMonitor(eval_env)
                    
                    print(f"[DEBUG] Round {round_idx + 1}: Evaluating vanilla policy...")
                    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
                    eval_msg = f"Vanilla Round {round_idx + 1}: Eval Reward: {mean_reward:.2f} ± {std_reward:.2f}"
                    print(eval_msg)
                    write_and_print_log(training_log, f"{eval_msg}\n", also_print=False)

                    # Record video (optional - skip if issues)
                    print(f"[DEBUG] Round {round_idx + 1}: Attempting to record vanilla agent video...")
                    video_path = os.path.join(VANILLA_LOG_PATH, f"vanilla_ppo_round_{round_idx + 1}.mp4")
                    try:
                        record_agent_video(eval_env, model, path=video_path, max_frames=500)
                    except Exception as e:
                        video_error_msg = f"Vanilla Round {round_idx + 1}: Video recording failed (skipping): {str(e)}"
                        print(f"[INFO] {video_error_msg}")
                        # Don't log video errors - just continue silently
                        print("[INFO] Continuing without video (this is normal)...")

                    # Collect trajectories for analysis
                    print(f"[DEBUG] Round {round_idx + 1}: Collecting vanilla trajectories...")
                    successful, unsuccessful = collect_trajectories(train_env, model, n_episodes=10, round_idx=round_idx)
                    
                    traj_msg = f"Vanilla Round {round_idx + 1}: Collected {len(successful)} successful, {len(unsuccessful)} unsuccessful trajectories"
                    print(traj_msg)
                    write_and_print_log(training_log, f"{traj_msg}\n", also_print=False)

                    # Trajectory summary
                    if successful and unsuccessful:
                        avg_success_len = np.mean([len(t) for t in successful])
                        avg_fail_len = np.mean([len(t) for t in unsuccessful])
                        summary = (
                            f"Vanilla Round {round_idx + 1} Trajectory Summary:\n"
                            f"Successful trajectories: {len(successful)}\n"
                            f"Unsuccessful trajectories: {len(unsuccessful)}\n"
                            f"Average length success: {avg_success_len:.2f}\n"
                            f"Average length fail: {avg_fail_len:.2f}\n"
                        )
                        print(f"[DEBUG] {summary}")
                        write_and_print_log(training_log, f"{summary}\n", also_print=False)
                    else:
                        insufficient_msg = f"Vanilla Round {round_idx + 1}: Insufficient trajectory data for analysis"
                        print(insufficient_msg)
                        write_and_print_log(training_log, f"{insufficient_msg}\n")

                    # Clean up environments
                    train_env.close()
                    eval_env.close()

                except Exception as e:
                    error_msg = f"Vanilla Round {round_idx + 1}: EXCEPTION OCCURRED: {str(e)}\n{traceback.format_exc()}"
                    print(f"[ERROR] {error_msg}")
                    write_and_print_log(training_log, f"ERROR: {error_msg}\n")
                    continue

            # Final completion message
            completion_msg = f"\nVanilla PPO training complete at {datetime.now().isoformat()}\n"
            print(completion_msg.strip())
            write_and_print_log(training_log, completion_msg, also_print=False)

    except Exception as e:
        fatal_error_msg = f"[FATAL ERROR] Failed to open training log file: {e}"
        print(fatal_error_msg)
        return

    print("[DEBUG] Vanilla training completed successfully")
    print(f"[DEBUG] Results saved to: {VANILLA_LOG_PATH}")
    print("[DEBUG] You can now compare vanilla PPO performance with LLM reward function performance")

if __name__ == "__main__":
    main()