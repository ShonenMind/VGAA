# Direct procgen inspection - bypasses gym compatibility issues

import numpy as np
from procgen import ProcgenEnv
import procgen

def test_direct_procgen():
    """Test procgen directly without gym"""
    print("="*60)
    print("TESTING DIRECT PROCGEN (NO GYM)")
    print("="*60)
    
    try:
        # Create environment directly
        print("Creating ProcgenEnv directly...")
        env = ProcgenEnv(
            num_envs=1, 
            env_name='coinrun', 
            num_levels=1,
            start_level=0
        )
        
        print("✓ Environment created successfully!")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Reset
        print("\nTesting reset...")
        obs = env.reset()
        print(f"✓ Reset successful!")
        print(f"Obs type: {type(obs)}")
        print(f"Obs keys: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}")
        
        if isinstance(obs, dict):
            for key, value in obs.items():
                print(f"  obs['{key}'] shape: {value.shape}, dtype: {value.dtype}")
                if key == 'rgb':
                    print(f"  RGB range: {value.min()} to {value.max()}")
        
        # Take steps
        for step in range(3):
            print(f"\n--- Direct Procgen Step {step + 1} ---")
            
            # Sample action (must be array for vectorized env)
            action = np.array([env.action_space.sample()])
            print(f"Action: {action}")
            
            # Step
            obs, reward, done, info = env.step(action)
            
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print(f"Info type: {type(info)}")
            print(f"Info: {info}")
            
            # Check info structure
            if isinstance(info, (list, tuple)) and len(info) > 0:
                first_info = info[0]
                print(f"First env info: {first_info}")
                if isinstance(first_info, dict):
                    print("Info keys:")
                    for key, value in first_info.items():
                        print(f"  '{key}': {value} (type: {type(value)})")
                        
                    # Check for velocity specifically
                    if 'velocity' in first_info:
                        print(f"🎉 VELOCITY FOUND: {first_info['velocity']}")
                    if 'x_pos' in first_info:
                        print(f"🎉 X_POS FOUND: {first_info['x_pos']}")
                    if 'coins_collected' in first_info:
                        print(f"🎉 COINS_COLLECTED FOUND: {first_info['coins_collected']}")
            
            elif isinstance(info, dict):
                print("Direct info dict:")
                for key, value in info.items():
                    print(f"  '{key}': {value} (type: {type(value)})")
                    
                # Check for velocity
                if 'velocity' in info:
                    print(f"🎉 VELOCITY FOUND: {info['velocity']}")
            
            if np.any(done):
                print("Episode ended")
                break
        
        env.close()
        return True
        
    except Exception as e:
        print(f"Error with direct procgen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_procgen_gym3():
    """Test the gym3 interface if available"""
    print("\n" + "="*60)
    print("TESTING PROCGEN GYM3 INTERFACE")
    print("="*60)
    
    try:
        from procgen import ProcgenGym3Env
        
        print("Creating ProcgenGym3Env...")
        env = ProcgenGym3Env(num=1, env_name='coinrun', num_levels=1)
        
        print("✓ Gym3 environment created!")
        
        # Reset
        result = env.observe()
        print(f"Initial observation: {type(result)}")
        
        # Take a step
        action = np.array([env.action_space.sample()])
        env.act(action)
        result = env.observe()
        
        print(f"After step observation: {type(result)}")
        print(f"Result keys: {result.keys() if hasattr(result, 'keys') else 'No keys'}")
        
        env.close()
        return True
        
    except ImportError:
        print("ProcgenGym3Env not available")
        return False
    except Exception as e:
        print(f"Error with gym3 interface: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_procgen_version():
    """Check what version of procgen we have"""
    print("="*60)
    print("CHECKING PROCGEN VERSION")
    print("="*60)
    
    try:
        import procgen
        print(f"Procgen module: {procgen}")
        
        # Try to get version
        if hasattr(procgen, '__version__'):
            print(f"Procgen version: {procgen.__version__}")
        else:
            print("No version attribute found")
        
        # Check what's available
        print("Available in procgen module:")
        for attr in dir(procgen):
            if not attr.startswith('_'):
                print(f"  {attr}")
        
        # Check for specific classes
        if hasattr(procgen, 'ProcgenEnv'):
            print("✓ ProcgenEnv available")
        if hasattr(procgen, 'ProcgenGym3Env'):
            print("✓ ProcgenGym3Env available")
            
    except ImportError as e:
        print(f"Could not import procgen: {e}")

def test_paint_vel_info():
    """Test if paint_vel_info option is available and works"""
    print("\n" + "="*60)
    print("TESTING PAINT_VEL_INFO OPTION")
    print("="*60)
    
    # Test with paint_vel_info=True
    print("Testing with paint_vel_info=True...")
    try:
        env = ProcgenEnv(
            num_envs=1, 
            env_name='coinrun', 
            num_levels=1,
            paint_vel_info=True
        )
        print("✓ paint_vel_info=True accepted!")
        
        obs = env.reset()
        print(f"Obs keys: {list(obs.keys())}")
        
        # Check if there are additional keys beyond 'rgb'
        for key, value in obs.items():
            print(f"  {key}: shape {getattr(value, 'shape', 'no shape')}")
        
        # Take a step and check info
        action = np.array([env.action_space.sample()])
        obs, reward, done, info = env.step(action)
        
        print(f"After step with paint_vel_info=True:")
        print(f"  Info: {info}")
        if isinstance(info, list) and len(info) > 0:
            first_info = info[0]
            print(f"  Info keys: {list(first_info.keys()) if isinstance(first_info, dict) else 'Not dict'}")
            if isinstance(first_info, dict):
                for key, value in first_info.items():
                    print(f"    '{key}': {value}")
        
        # Check if velocity info is painted on the RGB observation
        rgb_obs = obs['rgb'][0]  # Get first env's observation
        print(f"  RGB observation shape: {rgb_obs.shape}")
        print(f"  RGB top-left corner (where velocity might be painted):")
        print(f"    Top-left 10x10 pixel values: {rgb_obs[:10, :10, 0]}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ paint_vel_info=True failed: {e}")
        return False

def test_coinrun_specifically():
    """Test coinrun environment with various parameters"""
    print("\n" + "="*60)
    print("TESTING COINRUN SPECIFICALLY")
    print("="*60)
    
    test_configs = [
        {'num_envs': 1, 'env_name': 'coinrun', 'num_levels': 1},
        {'num_envs': 1, 'env_name': 'coinrun', 'num_levels': 1, 'start_level': 0},
        {'num_envs': 1, 'env_name': 'coinrun', 'num_levels': 0},  # Unlimited levels
        {'num_envs': 1, 'env_name': 'coinrun', 'num_levels': 1, 'paint_vel_info': True},  # NEW: Test velocity painting
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nCoinRun Test {i+1}: {config}")
        try:
            env = ProcgenEnv(**config)
            print("✓ Environment created")
            
            obs = env.reset()
            print(f"✓ Reset successful, obs type: {type(obs)}")
            if isinstance(obs, dict):
                print(f"  Obs keys: {list(obs.keys())}")
                for key, value in obs.items():
                    print(f"  obs['{key}'] shape: {getattr(value, 'shape', 'no shape')}")
            else:
                print(f"  Obs shape: {getattr(obs, 'shape', 'no shape')}")
            
            # Take one step to see info
            action = np.array([env.action_space.sample()])
            obs, reward, done, info = env.step(action)
            print(f"✓ Step successful")
            print(f"  Reward: {reward}")
            print(f"  Info type: {type(info)}")
            
            # Look for velocity info
            if isinstance(info, (list, tuple)) and len(info) > 0:
                if isinstance(info[0], dict):
                    keys = list(info[0].keys())
                    print(f"  Info keys: {keys}")
                    if 'velocity' in keys:
                        print(f"  🎉 VELOCITY: {info[0]['velocity']}")
            
            env.close()
            
        except Exception as e:
            print(f"✗ Failed: {e}")

if __name__ == "__main__":
    print("Starting Direct Procgen Inspection (Bypassing Gym)...")
    
    # Run tests in order
    check_procgen_version()
    
    # Test direct procgen interface
    direct_success = test_direct_procgen()
    
    # Test paint_vel_info specifically
    paint_vel_success = test_paint_vel_info()
    
    # Test gym3 interface
    gym3_success = test_procgen_gym3()
    
    # Test coinrun specifically
    test_coinrun_specifically()
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE")
    print("="*60)
    
    if direct_success:
        print("✅ Direct ProcgenEnv works!")
    if gym3_success:
        print("✅ ProcgenGym3Env works!")
    if paint_vel_success:
        print("✅ paint_vel_info option works!")
    else:
        print("❌ paint_vel_info option not available or doesn't work")
    
    print("\nNext steps:")
    print("- If velocity info was found, update your env_wrapper.py")
    print("- Use ProcgenEnv directly instead of gym.make()")
    print("- Consider using pixel extraction from adaptive_pixel_utils.py")