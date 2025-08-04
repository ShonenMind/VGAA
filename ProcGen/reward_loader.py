import math
import random
import numpy as np
import statistics
def load_reward_fn(reward_code_str):
    """
    Safely load the reward_fn function from a code string.
    Includes debugging if 'random' or other symbols are not defined.
    """
    if "random." in reward_code_str and "import random" not in reward_code_str:
        print("[DEBUG] Patching missing `import random`")
        reward_code_str = "import random\n" + reward_code_str

    if "math." in reward_code_str and "import math" not in reward_code_str:
        print("[DEBUG] Patching missing `import math`")
        reward_code_str = "import math\n" + reward_code_str

    if "np." in reward_code_str and "import numpy" not in reward_code_str:
        print("[DEBUG] Patching missing `import numpy`")
        reward_code_str = "import numpy as np\n" + reward_code_str

    with open("temp_reward_debug.py", "w") as f:
        f.write(reward_code_str)
        print("[DEBUG] Saved patched reward function to temp_reward_debug.py")

    local_vars = {}
    try:
        global_env = {
            "__builtins__": __builtins__,
            "random": random,
            "math": math,
            "np": np,
            "numpy": np,
            "statistics": statistics
        }
        exec(reward_code_str, global_env, local_vars)
    except Exception as e:
        print("[ERROR] Exception while loading reward function:", e)
        return None

    reward_fn = local_vars.get('reward_fn') or global_env.get('reward_fn')
    if reward_fn is None:
        print("[ERROR] `reward_fn` not found in loaded code.")
    return reward_fn
