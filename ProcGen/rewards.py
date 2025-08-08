import os
import re
from openai import OpenAI
from datetime import datetime

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def extract_code_block(response_text):
    """
    Extracts Python code from a markdown-style code block (```python ... ```).
    Falls back to best-effort heuristic if not found. Logs to file if all else fails.
    """
    match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: try to extract any code-looking region (just in case)
    lines = response_text.strip().splitlines()
    code_lines = [line for line in lines if line.strip() and not line.strip().lower().startswith("explanation")]
    if code_lines:
        return "\n".join(code_lines).strip()

    # If all extraction fails, log full response to file
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/invalid_llm_response_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write("Full LLM response:\n\n")
        f.write(response_text.strip() + "\n")
    raise ValueError(f"No valid Python code block found in LLM response. Logged to {filename}")


def _run_llm_prompt(system_msg, user_msg, temperature=0.7, max_tokens=300):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return extract_code_block(response.choices[0].message.content.strip())


def get_random_reward_fn():
    system_msg = (
        "You are an assistant that writes Python reward functions for reinforcement learning "
        "agents in the Procgen CoinRun environment. CoinRun is a visual environment where game state "
        "is encoded in 64x64x3 pixel observations, not in the info dictionary. The info dictionary "
        "only contains level metadata like 'prev_level_seed', 'level_seed', etc."
    )

    k_shot_example = """Here is an example of the format you should use:

```python
def reward_fn(state, action, info, original_reward=0):
    # Shape the original reward from the environment
    shaped_reward = original_reward * 2.0
    
    # Add exploration bonus
    exploration_bonus = 0.01
    
    # Add some small randomness for diversity
    import random
    noise = random.uniform(-0.005, 0.005)
    
    return shaped_reward + exploration_bonus + noise
```"""

    user_msg = f"""
Write a Python function named `reward_fn(state, action, info, original_reward=0)` that returns a float reward.

IMPORTANT CONSTRAINTS:
- The info dictionary ONLY contains: 'prev_level_seed', 'prev_level_complete', 'level_seed', 'TimeLimit.truncated'
- Game state (position, coins) is NOT available in info - it's encoded in the 64x64x3 pixel observation
- Use the original_reward parameter as your base - it contains the environment's built-in reward signal
- You can shape/scale the original reward and add bonuses for exploration, action diversity, etc.
- Do NOT try to access non-existent keys like 'x_pos', 'coins_collected', 'velocity'

AVAILABLE PIXEL EXTRACTION FUNCTIONS (optional, from adaptive_pixel_utils):
You can also use these functions to extract game state from the pixel observation:
- get_player_x_position(obs), which returns float (0-64) representing player's x-coordinate
- get_player_y_position(obs), which returns float (0-64) representing player's y-coordinate  
- count_coins_visible(obs), which returns float (0-5) representing estimated number of coins visible
- get_obstacle_density(obs), which returns float (0-1) representing density of obstacles/hazards
- estimate_progress(obs), which returns float (0-10) representing overall progress score
- get_ground_level(obs), which returns float representing Y-coordinate of ground/platforms
- is_player_on_ground(obs), which returns bool indicating if player is on ground
- calculate_pixel_diversity(obs), which returns float (0-1) representing visual complexity
- get_comprehensive_state(obs), which returns dict with all above metrics

To use these, you would: obs = state.get('obs', np.zeros((64, 64, 3))) and then call the functions.

Focus on:
- Scaling/shaping the original_reward
- Adding small exploration bonuses  
- Encouraging action diversity
- Using level completion (info.get('prev_level_complete', 0)) if helpful
- Adding small amounts of randomness for exploration

ONLY output one Python code block like ```python ... ``` and nothing else.

{k_shot_example}
"""

    return _run_llm_prompt(system_msg, user_msg, temperature=0.9)


def get_reactive_reward_fn(previous_code: str, trajectory_summary: str):
    system_msg = (
        "You are an assistant that writes Python reward functions for the Procgen CoinRun environment. "
        "CoinRun info only contains level metadata, not game state. Use the original_reward parameter."
    )

    user_msg = f"""
Your previous reward function did not work well and failed preference evaluation.

Previous reward function:

{previous_code}

Trajectory statistics:

{trajectory_summary}

Please fix the reward function. Write a new function named `reward_fn(state, action, info, original_reward=0)` that returns a float.

IMPORTANT:
- Use original_reward parameter as the base reward signal
- The info dictionary only contains level metadata, not game state
- Focus on reward shaping, exploration bonuses, and action diversity
- Do NOT use non-existent keys like 'x_pos', 'coins_collected', 'velocity'

AVAILABLE PIXEL EXTRACTION FUNCTIONS (optional, from adaptive_pixel_utils):
You can also use these functions to extract game state from the pixel observation:
- get_player_x_position(obs), which returns float (0-64) representing player's x-coordinate
- get_player_y_position(obs), which returns float (0-64) representing player's y-coordinate  
- count_coins_visible(obs), which returns float (0-5) representing estimated number of coins visible
- get_obstacle_density(obs), which returns float (0-1) representing density of obstacles/hazards
- estimate_progress(obs), which returns float (0-10) representing overall progress score
- is_player_on_ground(obs), which returns bool indicating if player is on ground
- calculate_pixel_diversity(obs), which returns float (0-1) representing visual complexity

To use these, you would: obs = state.get('obs', np.zeros((64, 64, 3))) and then call the functions.

Only output a single Python code block like ```python ... ```. Do not include any explanation or text outside the code block.
"""

    code = _run_llm_prompt(system_msg, user_msg)

    # Save to debug file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_reactive_reward_fn_{timestamp}.py"
    with open(filename, "w") as f:
        f.write(code)
    print(f"[DEBUG] Saved reactive reward function to {filename}")

    return code


def get_proactive_reward_fn(current_code: str):
    system_msg = (
        "You are an assistant that helps improve Python reward functions for the Procgen CoinRun environment. "
        "CoinRun info only contains level metadata. Use the original_reward parameter for reward shaping."
    )

    user_msg = f"""
The current reward function passed preference evaluation (TPE score ≥ 0.8).

Current reward function:

{current_code}

Would you like to optionally improve or refine this function to further optimize performance or explore alternate behaviors?
Write a new version of `reward_fn(state, action, info, original_reward=0)` if you want.

IMPORTANT:
- Use original_reward parameter as the base reward signal
- The info dictionary only contains level metadata, not game state  
- Focus on reward shaping, exploration bonuses, and action diversity

AVAILABLE PIXEL EXTRACTION FUNCTIONS (optional, from adaptive_pixel_utils):
You can also use these functions to extract game state from the pixel observation:
- get_player_x_position(obs), which returns float (0-64) representing player's x-coordinate
- get_player_y_position(obs), which returns float (0-64) representing player's y-coordinate  
- count_coins_visible(obs), which returns float (0-5) representing estimated number of coins visible
- get_obstacle_density(obs), which returns float (0-1) representing density of obstacles/hazards
- estimate_progress(obs), which returns float (0-10) representing overall progress score
- is_player_on_ground(obs), which returns bool indicating if player is on ground
- calculate_pixel_diversity(obs), which returns float (0-1) representing visual complexity
- get_movement_direction(obs, prev_obs), which returns float (-1 to 1) representing movement direction

To use these, you would: obs = state.get('obs', np.zeros((64, 64, 3))) and then call the functions.

Only output a single Python code block like ```python ... ```. No explanations.
"""

    code = _run_llm_prompt(system_msg, user_msg, temperature=0.9)

    # Save to debug file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_proactive_reward_fn_{timestamp}.py"
    with open(filename, "w") as f:
        f.write(code)
    print(f"[DEBUG] Saved proactive reward function to {filename}")

    return code


def should_proactively_revise(current_code: str) -> bool:
    system_msg = "You are a reinforcement learning evaluator helping decide whether to revise a reward function."

    user_msg = f"""
The current reward function passed preference evaluation (TPE score ≥ 0.8).
Do you want to revise it to further improve it?
Respond with only one word: "yes" or "no".

Current reward function:

{current_code}
"""

    reply = _run_llm_prompt(system_msg, user_msg, temperature=0.3, max_tokens=5).lower()
    return reply in ["yes", "y"]


if __name__ == "__main__":
    code = get_random_reward_fn()
    print("=== Random initial reward function ===\n")
    print(code)

'''import os
import re
from openai import OpenAI
from datetime import datetime

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def extract_code_block(response_text):
    """
    Extracts Python code from a markdown-style code block (```python ... ```).
    Falls back to best-effort heuristic if not found. Logs to file if all else fails.
    """
    match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: try to extract any code-looking region (just in case)
    lines = response_text.strip().splitlines()
    code_lines = [line for line in lines if line.strip() and not line.strip().lower().startswith("explanation")]
    if code_lines:
        return "\n".join(code_lines).strip()

    # If all extraction fails, log full response to file
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/invalid_llm_response_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write("Full LLM response:\n\n")
        f.write(response_text.strip() + "\n")
    raise ValueError(f"No valid Python code block found in LLM response. Logged to {filename}")


def _run_llm_prompt(system_msg, user_msg, temperature=0.7, max_tokens=300):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return extract_code_block(response.choices[0].message.content.strip())


def get_random_reward_fn():
    system_msg = (
        "You are an assistant that writes Python reward functions for reinforcement learning "
        "agents in the Procgen CoinRun environment. CoinRun is a visual environment where game state "
        "is encoded in 64x64x3 pixel observations, not in the info dictionary. The info dictionary "
        "only contains level metadata like 'prev_level_seed', 'level_seed', etc."
    )

    k_shot_example = """Here is an example of the format you should use:

```python
def reward_fn(state, action, info, original_reward=0):
    # Shape the original reward from the environment
    shaped_reward = original_reward * 2.0
    
    # Add exploration bonus
    exploration_bonus = 0.01
    
    # Add some small randomness for diversity
    import random
    noise = random.uniform(-0.005, 0.005)
    
    return shaped_reward + exploration_bonus + noise
```"""

    user_msg = f"""
Write a Python function named `reward_fn(state, action, info, original_reward=0)` that returns a float reward.

IMPORTANT CONSTRAINTS:
- The info dictionary ONLY contains: 'prev_level_seed', 'prev_level_complete', 'level_seed', 'TimeLimit.truncated'
- Game state (position, coins) is NOT available in info - it's encoded in the 64x64x3 pixel observation
- Use the original_reward parameter as your base - it contains the environment's built-in reward signal
- You can shape/scale the original reward and add bonuses for exploration, action diversity, etc.
- Do NOT try to access non-existent keys like 'x_pos', 'coins_collected', 'velocity'

Focus on:
- Scaling/shaping the original_reward
- Adding small exploration bonuses  
- Encouraging action diversity
- Using level completion (info.get('prev_level_complete', 0)) if helpful
- Adding small amounts of randomness for exploration

ONLY output one Python code block like ```python ... ``` and nothing else.

{k_shot_example}
"""

    return _run_llm_prompt(system_msg, user_msg, temperature=0.9)


def get_reactive_reward_fn(previous_code: str, trajectory_summary: str):
    system_msg = (
        "You are an assistant that writes Python reward functions for the Procgen CoinRun environment. "
        "CoinRun info only contains level metadata, not game state. Use the original_reward parameter."
    )

    user_msg = f"""
Your previous reward function did not work well and failed preference evaluation.

Previous reward function:

{previous_code}

Trajectory statistics:

{trajectory_summary}

Please fix the reward function. Write a new function named `reward_fn(state, action, info, original_reward=0)` that returns a float.

IMPORTANT:
- Use original_reward parameter as the base reward signal
- The info dictionary only contains level metadata, not game state
- Focus on reward shaping, exploration bonuses, and action diversity
- Do NOT use non-existent keys like 'x_pos', 'coins_collected', 'velocity'

Only output a single Python code block like ```python ... ```. Do not include any explanation or text outside the code block.
"""

    code = _run_llm_prompt(system_msg, user_msg)

    # Save to debug file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_reactive_reward_fn_{timestamp}.py"
    with open(filename, "w") as f:
        f.write(code)
    print(f"[DEBUG] Saved reactive reward function to {filename}")

    return code


def get_proactive_reward_fn(current_code: str):
    system_msg = (
        "You are an assistant that helps improve Python reward functions for the Procgen CoinRun environment. "
        "CoinRun info only contains level metadata. Use the original_reward parameter for reward shaping."
    )

    user_msg = f"""
The current reward function passed preference evaluation (TPE score ≥ 0.8).

Current reward function:

{current_code}

Would you like to optionally improve or refine this function to further optimize performance or explore alternate behaviors?
Write a new version of `reward_fn(state, action, info, original_reward=0)` if you want.

IMPORTANT:
- Use original_reward parameter as the base reward signal
- The info dictionary only contains level metadata, not game state  
- Focus on reward shaping, exploration bonuses, and action diversity

Only output a single Python code block like ```python ... ```. No explanations.
"""

    code = _run_llm_prompt(system_msg, user_msg, temperature=0.9)

    # Save to debug file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_proactive_reward_fn_{timestamp}.py"
    with open(filename, "w") as f:
        f.write(code)
    print(f"[DEBUG] Saved proactive reward function to {filename}")

    return code


def should_proactively_revise(current_code: str) -> bool:
    system_msg = "You are a reinforcement learning evaluator helping decide whether to revise a reward function."

    user_msg = f"""
The current reward function passed preference evaluation (TPE score ≥ 0.8).
Do you want to revise it to further improve it?
Respond with only one word: "yes" or "no".

Current reward function:

{current_code}
"""

    reply = _run_llm_prompt(system_msg, user_msg, temperature=0.3, max_tokens=5).lower()
    return reply in ["yes", "y"]


if __name__ == "__main__":
    code = get_random_reward_fn()
    print("=== Random initial reward function ===\n")
    print(code)'''