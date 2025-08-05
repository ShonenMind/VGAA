import math
import random
import numpy as np
import statistics
import os
from datetime import datetime

def load_reward_fn(reward_code_str):
    os.makedirs("logs", exist_ok=True)  # Ensure logs dir exists
    
    # Log input snippet for debugging
    with open("logs/debug_load_reward_fn_input.log", "a") as f:
        f.write(f"\n--- Loading Reward Function at {datetime.now()} ---\n")
        f.write(reward_code_str[:1000] + "\n")  # Log first 1000 chars
    
    # 1. Auto-fix missing imports
    if "random." in reward_code_str and "import random" not in reward_code_str:
        reward_code_str = "import random\n" + reward_code_str
    if "math." in reward_code_str and "import math" not in reward_code_str:
        reward_code_str = "import math\n" + reward_code_str
    if "np." in reward_code_str and "import numpy" not in reward_code_str:
        reward_code_str = "import numpy as np\n" + reward_code_str

    # 2. Create a dedicated namespace
    global_env = {
        "__builtins__": __builtins__,
        "math": math,
        "random": random,
        "np": np,
        "statistics": statistics
    }

    # 3. Use a temp variable to capture the function
    exec_wrapper = f"""
__temp_reward_fn = None
{reward_code_str}
__temp_reward_fn = reward_fn  # Capture the function
"""
    try:
        exec(exec_wrapper, global_env)
        reward_fn = global_env.get("__temp_reward_fn")

        # Log whether the function was successfully extracted
        with open("logs/debug_load_reward_fn_output.log", "a") as f:
            status = "NOT None" if reward_fn else "None"
            f.write(f"[{datetime.now()}] reward_fn is {status}\n")

        if reward_fn is None:
            with open("logs/reward_load_errors.log", "a") as f:
                f.write(f"[{datetime.now()}] Failed to load reward function. Code snippet:\n{reward_code_str[:500]}\n")
            return None
            
        return reward_fn
        
    except Exception as e:
        with open("logs/reward_load_errors.log", "a") as f:
            f.write(f"[{datetime.now()}] Load error: {str(e)}\nCode snippet:\n{reward_code_str[:500]}\n")
        return None
