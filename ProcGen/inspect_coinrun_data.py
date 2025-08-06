#inspect what data is available in CoinRun environment

import gym
import numpy as np
import procgen
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env_wrapper import ProcgenCoinRunEnvWrapper

def inspect_single_environment():
    """Inspect a single (non-vectorized) environment"""
    print("="*60)
    print("INSPECTING SINGLE ENVIRONMENT")
    print("="*60)
    
    # Create single environment
    env = ProcgenCoinRunEnvWrapper(reward_code=None)  # No custom reward
    
    # Reset and get initial observation
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation type: {type(obs)}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Take a few random steps to see what info we get
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"\n--- Step {step + 1} ---")
        print(f"Action: {action} (type: {type(action)})")
        print(f"Reward: {reward} (type: {type(reward)})")
        print(f"Done: {done} (type: {type(done)})")
        print(f"Info type: {type(info)}")
        
        if isinstance(info, dict):
            print("Info keys and values:")
            for key, value in info.items():
                print(f"  info['{key}'] = {value} (type: {type(value)})")
        else:
            print(f"Info is not a dict: {info}")
        
        print(f"Observation shape after step: {obs.shape}")
        
        if done:
            print("Episode ended, resetting...")
            obs = env.reset()
            break
    
    env.close()

def inspect_vectorized_environment():
    """Inspect vectorized environment (like what you're using in training)"""
    print("\n" + "="*60)
    print("INSPECTING VECTORIZED ENVIRONMENT (4 parallel envs)")
    print("="*60)
    
    # Create vectorized environment like in your training
    env = DummyVecEnv([lambda: ProcgenCoinRunEnvWrapper(reward_code=None) for _ in range(4)])
    
    # Reset and get initial observations
    obs = env.reset()
    print(f"Vectorized obs shape: {obs.shape}")
    print(f"Vectorized obs type: {type(obs)}")
    
    # Take a few steps
    for step in range(3):
        actions = [env.action_space.sample() for _ in range(4)]  # 4 random actions
        obs, rewards, dones, infos = env.step(actions)
        
        print(f"\n--- Vectorized Step {step + 1} ---")
        print(f"Actions: {actions} (type: {type(actions)})")
        print(f"Rewards: {rewards} (type: {type(rewards)})")
        print(f"Dones: {dones} (type: {type(dones)})")
        print(f"Infos type: {type(infos)}, length: {len(infos) if hasattr(infos, '__len__') else 'N/A'}")
        
        # Inspect each environment's info
        if isinstance(infos, (list, tuple)):
            for i, info in enumerate(infos):
                print(f"  Environment {i} info:")
                if isinstance(info, dict):
                    for key, value in info.items():
                        print(f"    info['{key}'] = {value} (type: {type(value)})")
                else:
                    print(f"    Info is not a dict: {info}")
        
        if any(dones):
            print("At least one environment ended")
            break
    
    env.close()

def inspect_with_trained_model():
    """Use a trained model to get more realistic data"""
    print("\n" + "="*60)
    print("INSPECTING WITH TRAINED MODEL (if available)")
    print("="*60)
    
    try:
        # Try to load your trained model
        model_files = [
            "ppo_model_round_1",
            "ppo_model_round_2", 
            "ppo_model_round_3",
            "ppo_procgen_model"
        ]
        
        model = None
        for model_file in model_files:
            try:
                model = PPO.load(model_file)
                print(f"Loaded model: {model_file}")
                break
            except:
                continue
        
        if model is None:
            print("No trained model found, skipping model-based inspection")
            return
        
        # Create environment
        env = DummyVecEnv([lambda: ProcgenCoinRunEnvWrapper(reward_code=None)])
        obs = env.reset()
        
        print("Using trained model to generate actions...")
        
        for step in range(5):
            # Use model to predict action (more realistic than random)
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            print(f"\n--- Model Step {step + 1} ---")
            print(f"Model action: {action}")
            print(f"Rewards: {rewards}")
            print(f"Dones: {dones}")
            
            # Focus on first environment's info
            if isinstance(infos, (list, tuple)) and len(infos) > 0:
                info = infos[0]
                print("First env info:")
                if isinstance(info, dict):
                    for key, value in info.items():
                        print(f"  info['{key}'] = {value}")
            
            if any(dones):
                break
        
        env.close()
        
    except Exception as e:
        print(f"Error during model-based inspection: {e}")

def inspect_procgen_directly():
    """Inspect the base procgen environment directly"""
    print("\n" + "="*60)
    print("INSPECTING BASE PROCGEN ENVIRONMENT")
    print("="*60)
    
    try:
        env = gym.make('procgen:procgen-coinrun-v0', num_levels=1)
        obs = env.reset()
        
        print(f"Base procgen obs shape: {obs.shape}")
        print(f"Base procgen action space: {env.action_space}")
        
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            print(f"\n--- Base Procgen Step {step + 1} ---")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print(f"Info: {info}")
            if isinstance(info, dict):
                for key, value in info.items():
                    print(f"  base_info['{key}'] = {value}")
            
            if done:
                break
        
        env.close()
        
    except Exception as e:
        print(f"Error inspecting base procgen: {e}")

if __name__ == "__main__":
    print("Starting CoinRun Data Inspection...")
    
    # Run all inspections
    inspect_procgen_directly()
    inspect_single_environment()
    inspect_vectorized_environment()
    inspect_with_trained_model()
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE")
    