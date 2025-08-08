#!/usr/bin/env python3, written w claude
"""
Test adaptive_pixel_utils.py functions with real CoinRun environment
no openAI key needed; just tests the pixel extraction directly
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Test if we can import our pixel utilities
try:
    from adaptive_pixel_utils import (
        get_player_x_position, get_player_y_position, count_coins_visible,
        get_obstacle_density, estimate_progress, get_ground_level,
        is_player_on_ground, calculate_pixel_diversity, get_comprehensive_state
    )
    print("✅ Successfully imported adaptive_pixel_utils functions")
except ImportError as e:
    print(f"❌ Could not import adaptive_pixel_utils: {e}")
    print("Make sure adaptive_pixel_utils.py is in the same directory")
    exit(1)

def test_single_functions(obs, step_num):
    """Test each function individually and print results"""
    print(f"\n--- Testing Functions on Step {step_num} ---")
    
    try:
        player_x = get_player_x_position(obs)
        print(f"Player X position: {player_x:.2f}")
    except Exception as e:
        print(f"Error in get_player_x_position: {e}")
    
    try:
        player_y = get_player_y_position(obs)
        print(f"Player Y position: {player_y:.2f}")
    except Exception as e:
        print(f"Error in get_player_y_position: {e}")
    
    try:
        coins = count_coins_visible(obs)
        print(f"Coins visible: {coins:.2f}")
    except Exception as e:
        print(f"Error in count_coins_visible: {e}")
    
    try:
        obstacles = get_obstacle_density(obs)
        print(f"Obstacle density: {obstacles:.3f}")
    except Exception as e:
        print(f"Error in get_obstacle_density: {e}")
    
    try:
        progress = estimate_progress(obs)
        print(f"Progress estimate: {progress:.2f}")
    except Exception as e:
        print(f"Error in estimate_progress: {e}")
    
    try:
        ground = get_ground_level(obs)
        print(f"Ground level: {ground:.2f}")
    except Exception as e:
        print(f"Error in get_ground_level: {e}")
    
    try:
        on_ground = is_player_on_ground(obs)
        print(f"Player on ground: {on_ground}")
    except Exception as e:
        print(f"Error in is_player_on_ground: {e}")
    
    try:
        diversity = calculate_pixel_diversity(obs)
        print(f"Pixel diversity: {diversity:.3f}")
    except Exception as e:
        print(f"Error in calculate_pixel_diversity: {e}")

def test_comprehensive_state(obs, step_num):
    """Test the comprehensive state function"""
    try:
        state = get_comprehensive_state(obs)
        print(f"\n--- Comprehensive State (Step {step_num}) ---")
        for key, value in state.items():
            print(f"  {key}: {value}")
        return state
    except Exception as e:
        print(f"Error in get_comprehensive_state: {e}")
        return {}

def create_simple_reward_function():
    """Create a simple reward function to test (no OpenAI needed)"""
    def test_reward_fn(state, action, info, original_reward=0):
        """Simple test reward function using pixel extraction"""
        obs = state.get('obs', np.zeros((64, 64, 3)))
        
        try:
            # Test pixel extraction
            player_x = get_player_x_position(obs)
            coins = count_coins_visible(obs)
            progress = estimate_progress(obs)
            
            # Simple reward calculation
            position_reward = player_x * 0.1
            coin_reward = coins * 2.0
            progress_reward = progress * 0.2
            
            # Metadata
            level_complete = info.get('prev_level_complete', 0)
            completion_bonus = level_complete * 10.0
            
            total_reward = original_reward + position_reward + coin_reward + progress_reward + completion_bonus
            
            print(f"[REWARD] Original: {original_reward:.2f}, Position: {position_reward:.2f}, "
                  f"Coin: {coin_reward:.2f}, Progress: {progress_reward:.2f}, "
                  f"Completion: {completion_bonus:.2f}, Total: {total_reward:.2f}")
            
            return total_reward
            
        except Exception as e:
            print(f"[REWARD ERROR] {e}")
            return original_reward
    
    return test_reward_fn

def visualize_observation(obs, step_num, save_dir="logs"):
    """Save observation image for manual inspection"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(obs)
        plt.title(f"CoinRun Observation - Step {step_num}")
        plt.axis('off')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/coinrun_obs_step_{step_num}_{timestamp}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"💾 Saved observation to {filename}")
        
    except Exception as e:
        print(f"Error saving visualization: {e}")

def run_pixel_extraction_test():
    """Main test function"""
    print("="*60)
    print("TESTING ADAPTIVE_PIXEL_UTILS.PY")
    print("="*60)
    
    # Create environment
    try:
        env = gym.make('procgen:procgen-coinrun-v0', num_levels=1, start_level=0)
        print("✅ Successfully created CoinRun environment")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return
    
    # Create test reward function
    reward_fn = create_simple_reward_function()
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation type: {type(obs)}")
    print(f"Pixel value range: {obs.min()} - {obs.max()}")
    
    # Test functions on initial observation
    test_single_functions(obs, 0)
    comprehensive_state = test_comprehensive_state(obs, 0)
    visualize_observation(obs, 0)
    
    # Test reward function
    print(f"\n--- Testing Reward Function ---")
    test_reward = reward_fn({'obs': obs}, 0, {}, 0.0)
    
    # Take several steps and test
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"\n" + "="*40)
        print(f"STEP {step + 1}")
        print(f"Action: {action}, Original Reward: {reward}, Done: {done}")
        print("="*40)
        
        # Test all functions
        test_single_functions(obs, step + 1)
        comprehensive_state = test_comprehensive_state(obs, step + 1)
        
        # Test reward function
        state_dict = {'obs': obs}
        test_reward = reward_fn(state_dict, action, info, reward)
        
        # Save observation for first few steps
        if step < 3:
            visualize_observation(obs, step + 1)
        
        if done:
            print("Episode ended, resetting...")
            obs = env.reset()
    
    env.close()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("Check the logs/ directory for saved observation images")
    print("Look at the output above to see if pixel extraction is working")

if __name__ == "__main__":
    run_pixel_extraction_test()