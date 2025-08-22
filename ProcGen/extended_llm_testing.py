"""
extended_llm_training.py

PURPOSE:
This file exists to test the hypothesis that LLM-generated reward functions perform poorly 
because they don't get enough training time to be effective. The main training loop 
(main.py) only trains for 100K timesteps per iteration and frequently changes reward 
functions, creating a "moving target" problem where agents never fully adapt.

This experiment:
1. Generates a single LLM reward function (no iterations/changes)
2. Trains an agent for 1M timesteps with that fixed reward function  
3. Provides detailed performance tracking and comparison with vanilla PPO
4. Tests whether extended training time allows LLM rewards to become effective

HYPOTHESIS:
LLM reward functions will show competitive performance when given sufficient 
training time (1M timesteps) compared to vanilla PPO, and the poor performance 
in main.py is due to insufficient training rather than poor reward design.
"""

import os
import numpy as np
import traceback
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# Import existing functions
from env_wrapper import ProcgenCoinRunEnvWrapper
from rewards import get_random_reward_fn
from tpe import load_reward_fn, collect_trajectories, trajectory_preference_evaluation
from vanilla import VanillaProcgenEnv

# Extended training configuration
EXTENDED_TIMESTEPS = 1_000_000  # 1M timesteps vs 100K in main.py
EVAL_FREQUENCY = 50_000         # Evaluate every 50K timesteps
FINAL_EVAL_EPISODES = 100       # More thorough final evaluation

class PerformanceTrackingCallback(BaseCallback):
    """Custom callback to track performance metrics during training"""
    
    def __init__(self, eval_env, eval_freq=50000, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_std = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, 
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            
            self.evaluations_timesteps.append(self.n_calls)
            self.evaluations_results.append(mean_reward)
            self.evaluations_std.append(std_reward)
            
            if self.verbose > 0:
                print(f"Eval at {self.n_calls} timesteps: {mean_reward:.2f} ± {std_reward:.2f}")
                
        return True

def create_performance_logs_dir():
    """Create directory for extended training logs"""
    log_dir = "logs/extended_training"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def save_performance_plot(callback_llm, callback_vanilla, log_dir):
    """Create comparison plot of LLM vs Vanilla training curves"""
    plt.figure(figsize=(12, 8))
    
    # Plot LLM performance
    if callback_llm.evaluations_timesteps:
        plt.errorbar(
            callback_llm.evaluations_timesteps, 
            callback_llm.evaluations_results,
            yerr=callback_llm.evaluations_std,
            label='LLM Reward Function',
            color='blue',
            alpha=0.7
        )
    
    # Plot Vanilla performance  
    if callback_vanilla.evaluations_timesteps:
        plt.errorbar(
            callback_vanilla.evaluations_timesteps,
            callback_vanilla.evaluations_results, 
            yerr=callback_vanilla.evaluations_std,
            label='Vanilla PPO',
            color='orange',
            alpha=0.7
        )
    
    plt.xlabel('Training Timesteps')
    plt.ylabel('Mean Evaluation Reward')
    plt.title('Extended Training: LLM Rewards vs Vanilla PPO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(log_dir, 'training_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved training comparison to {plot_path}")

def write_summary_report(reward_code, llm_results, vanilla_results, log_dir):
    """Write detailed summary report of the experiment"""
    report_path = os.path.join(log_dir, 'experiment_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("EXTENDED LLM TRAINING EXPERIMENT SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Experiment Date: {datetime.now().isoformat()}\n")
        f.write(f"Training Timesteps: {EXTENDED_TIMESTEPS:,}\n")
        f.write(f"Evaluation Frequency: {EVAL_FREQUENCY:,}\n")
        f.write(f"Final Evaluation Episodes: {FINAL_EVAL_EPISODES}\n\n")
        
        f.write("HYPOTHESIS BEING TESTED:\n")
        f.write("LLM reward functions perform poorly in main.py due to insufficient\n")
        f.write("training time (100K timesteps) rather than poor reward design.\n")
        f.write("This experiment tests performance with 1M timesteps.\n\n")
        
        f.write("LLM REWARD FUNCTION USED:\n")
        f.write("-" * 40 + "\n")
        f.write(reward_code + "\n\n")
        
        f.write("FINAL RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"LLM Reward Final Performance: {llm_results['mean']:.2f} ± {llm_results['std']:.2f}\n")
        f.write(f"Vanilla PPO Final Performance: {vanilla_results['mean']:.2f} ± {vanilla_results['std']:.2f}\n")
        
        # Calculate performance difference
        diff = llm_results['mean'] - vanilla_results['mean']
        pct_diff = (diff / vanilla_results['mean']) * 100 if vanilla_results['mean'] != 0 else 0
        
        f.write(f"Difference: {diff:+.2f} ({pct_diff:+.1f}%)\n\n")
        
        # Determine outcome
        if diff > vanilla_results['std']:  # Significant improvement
            f.write("CONCLUSION: LLM reward function OUTPERFORMS vanilla PPO\n")
            f.write("HYPOTHESIS: SUPPORTED - Extended training enables LLM rewards\n")
        elif abs(diff) <= max(llm_results['std'], vanilla_results['std']):  # Similar performance
            f.write("CONCLUSION: LLM reward function performs SIMILARLY to vanilla PPO\n") 
            f.write("HYPOTHESIS: PARTIALLY SUPPORTED - Extended training helps, but limited benefit\n")
        else:  # Still underperforms
            f.write("CONCLUSION: LLM reward function still UNDERPERFORMS vanilla PPO\n")
            f.write("HYPOTHESIS: NOT SUPPORTED - Training time is not the main issue\n")
        
    print(f"[REPORT] Saved experiment summary to {report_path}")

def run_extended_llm_experiment():
    """Main function to run the extended training experiment"""
    print("="*80)
    print("EXTENDED LLM TRAINING EXPERIMENT")
    print("="*80)
    print(f"Training for {EXTENDED_TIMESTEPS:,} timesteps (vs 100K in main.py)")
    print(f"Testing hypothesis: LLM rewards need more training time to be effective")
    print()
    
    # Create logging directory
    log_dir = create_performance_logs_dir()
    
    # Generate single LLM reward function (no iterations)
    print("[STEP 1] Generating LLM reward function...")
    try:
        reward_code = get_random_reward_fn()
        print("✓ LLM reward function generated successfully")
        print("Reward function preview:")
        print(reward_code[:200] + "..." if len(reward_code) > 200 else reward_code)
        
        # Test loading the reward function
        reward_fn = load_reward_fn(reward_code)
        if reward_fn is None:
            print("✗ FATAL: Generated reward function failed to load")
            return
        print("✓ Reward function loads successfully")
        
    except Exception as e:
        print(f"✗ FATAL: Failed to generate LLM reward function: {e}")
        return
    
    # Save reward function to logs
    with open(os.path.join(log_dir, 'llm_reward_function.py'), 'w') as f:
        f.write(reward_code)
    
    print(f"\n[STEP 2] Training with LLM reward for {EXTENDED_TIMESTEPS:,} timesteps...")
    
    # Train with LLM reward function
    try:
        # Create environments
        llm_train_env = DummyVecEnv([lambda: ProcgenCoinRunEnvWrapper(reward_code=reward_code) for _ in range(4)])
        llm_train_env = VecMonitor(llm_train_env)
        
        llm_eval_env = DummyVecEnv([lambda: ProcgenCoinRunEnvWrapper(reward_code=reward_code)])
        llm_eval_env = VecMonitor(llm_eval_env)
        
        # Create callback for performance tracking
        llm_callback = PerformanceTrackingCallback(
            eval_env=llm_eval_env,
            eval_freq=EVAL_FREQUENCY,
            n_eval_episodes=10
        )
        
        # Train LLM model
        llm_model = PPO(
            "CnnPolicy",
            llm_train_env,
            verbose=1,
            n_steps=256,
            batch_size=1024,
            n_epochs=4,
            learning_rate=5e-4,
            ent_coef=0.01,
        )
        
        llm_model.learn(total_timesteps=EXTENDED_TIMESTEPS, callback=llm_callback)
        
        # Save model
        llm_model_path = os.path.join(log_dir, 'llm_model_extended')
        llm_model.save(llm_model_path)
        print(f"✓ LLM model saved to {llm_model_path}")
        
        # Final evaluation
        print(f"[EVALUATION] Final LLM evaluation with {FINAL_EVAL_EPISODES} episodes...")
        llm_mean, llm_std = evaluate_policy(llm_model, llm_eval_env, n_eval_episodes=FINAL_EVAL_EPISODES)
        llm_results = {'mean': llm_mean, 'std': llm_std}
        print(f"✓ LLM Final Performance: {llm_mean:.2f} ± {llm_std:.2f}")
        
        # Clean up
        llm_train_env.close()
        llm_eval_env.close()
        
    except Exception as e:
        print(f"✗ ERROR during LLM training: {e}")
        print(traceback.format_exc())
        return
    
    print(f"\n[STEP 3] Training vanilla PPO for {EXTENDED_TIMESTEPS:,} timesteps (baseline)...")
    
    # Train vanilla PPO for comparison
    try:
        # Create vanilla environments
        vanilla_train_env = DummyVecEnv([lambda: VanillaProcgenEnv() for _ in range(4)])
        vanilla_train_env = VecMonitor(vanilla_train_env)
        
        vanilla_eval_env = DummyVecEnv([lambda: VanillaProcgenEnv()])
        vanilla_eval_env = VecMonitor(vanilla_eval_env)
        
        # Create callback for performance tracking
        vanilla_callback = PerformanceTrackingCallback(
            eval_env=vanilla_eval_env,
            eval_freq=EVAL_FREQUENCY,
            n_eval_episodes=10
        )
        
        # Train vanilla model
        vanilla_model = PPO(
            "CnnPolicy",
            vanilla_train_env,
            verbose=1,
            n_steps=256,
            batch_size=1024,
            n_epochs=4,
            learning_rate=5e-4,
            ent_coef=0.01,
        )
        
        vanilla_model.learn(total_timesteps=EXTENDED_TIMESTEPS, callback=vanilla_callback)
        
        # Save model
        vanilla_model_path = os.path.join(log_dir, 'vanilla_model_extended')
        vanilla_model.save(vanilla_model_path)
        print(f"✓ Vanilla model saved to {vanilla_model_path}")
        
        # Final evaluation
        print(f"[EVALUATION] Final vanilla evaluation with {FINAL_EVAL_EPISODES} episodes...")
        vanilla_mean, vanilla_std = evaluate_policy(vanilla_model, vanilla_eval_env, n_eval_episodes=FINAL_EVAL_EPISODES)
        vanilla_results = {'mean': vanilla_mean, 'std': vanilla_std}
        print(f"✓ Vanilla Final Performance: {vanilla_mean:.2f} ± {vanilla_std:.2f}")
        
        # Clean up
        vanilla_train_env.close()
        vanilla_eval_env.close()
        
    except Exception as e:
        print(f"✗ ERROR during vanilla training: {e}")
        print(traceback.format_exc())
        return
    
    print(f"\n[STEP 4] Analyzing results and generating report...")
    
    # Create performance comparison plot
    save_performance_plot(llm_callback, vanilla_callback, log_dir)
    
    # Generate summary report
    write_summary_report(reward_code, llm_results, vanilla_results, log_dir)
    
    # Print final comparison
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"LLM Reward Performance:    {llm_results['mean']:8.2f} ± {llm_results['std']:.2f}")
    print(f"Vanilla PPO Performance:   {vanilla_results['mean']:8.2f} ± {vanilla_results['std']:.2f}")
    
    diff = llm_results['mean'] - vanilla_results['mean']
    pct_diff = (diff / vanilla_results['mean']) * 100 if vanilla_results['mean'] != 0 else 0
    print(f"Difference:                {diff:+8.2f} ({pct_diff:+.1f}%)")
    
    if diff > vanilla_results['std']:
        print("\n HYPOTHESIS SUPPORTED: LLM rewards outperform with extended training!")
    elif abs(diff) <= max(llm_results['std'], vanilla_results['std']):
        print("\n HYPOTHESIS PARTIALLY SUPPORTED: Similar performance with extended training")
    else:
        print("\n HYPOTHESIS NOT SUPPORTED: LLM rewards still underperform")
    
    print(f"\nDetailed results saved to: {log_dir}/")
    print("="*80)

if __name__ == "__main__":
    run_extended_llm_experiment()