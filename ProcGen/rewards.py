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
        "is encoded in 64x64x3 pixel observations. The info dictionary contains level metadata and "
        "velocity information extracted from the environment."
    )

    k_shot_example = """Here is an example of the format you should use:

```python
def reward_fn(state, action, info, original_reward=0):
    # Get velocity information
    vel_x = info.get('velocity_x', 0)
    vel_y = info.get('velocity_y', 0)
    
    # Shape the original reward from the environment
    shaped_reward = original_reward * 2.0
    
    # Reward rightward movement
    movement_bonus = max(vel_x * 0.1, 0)  # Only reward rightward
    
    # Small penalty for excessive vertical movement
    stability_penalty = abs(vel_y) * 0.05
    
    # Add exploration bonus
    exploration_bonus = 0.01
    
    return shaped_reward + movement_bonus + exploration_bonus - stability_penalty
```"""

    user_msg = f"""
Write a Python function named `reward_fn(state, action, info, original_reward=0)` that returns a float reward.

IMPORTANT CONSTRAINTS:
- The info dictionary now contains velocity information: 'velocity_x', 'velocity_y', 'velocity_magnitude'
- velocity_x: horizontal velocity (positive = moving right, negative = moving left)
- velocity_y: vertical velocity (positive = moving up, negative = moving down)  
- velocity_magnitude: overall speed regardless of direction
- The info dictionary also contains level metadata: 'prev_level_seed', 'prev_level_complete', 'level_seed'
- Use the original_reward parameter as your base - it contains the environment's built-in reward signal
- You can shape/scale the original reward and add bonuses based on velocity and movement patterns

VELOCITY USAGE EXAMPLES:
```python
def reward_fn(state, action, info, original_reward=0):
    vel_x = info.get('velocity_x', 0)
    vel_y = info.get('velocity_y', 0)
    speed = info.get('velocity_magnitude', 0)
    
    # Reward efficient rightward movement
    movement_reward = max(vel_x * 0.1, 0)
    
    # Penalize excessive jumping/falling
    stability_penalty = abs(vel_y) * 0.03
    
    # Bonus for maintaining good speed
    speed_bonus = min(speed * 0.05, 0.2)
    
    # Level completion bonus
    completion_bonus = info.get('prev_level_complete', 0) * 1.0
    
    return original_reward + movement_reward + speed_bonus - stability_penalty + completion_bonus
```

Focus on:
- Using velocity to encourage efficient movement patterns
- Rewarding or penalizing specific movement behaviors (rightward good, excessive jumping bad)
- Balancing speed vs stability
- Using original_reward as the foundation
- Adding small exploration bonuses

ONLY output one Python code block like ```python ... ``` and nothing else.

{k_shot_example}
"""

    return _run_llm_prompt(system_msg, user_msg, temperature=0.9)


def get_reactive_reward_fn(previous_code: str, trajectory_summary: str):
    system_msg = (
        "You are an assistant that writes Python reward functions for the Procgen CoinRun environment. "
        "The info dictionary contains level metadata and velocity information (velocity_x, velocity_y, velocity_magnitude)."
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
- The info dictionary contains velocity information: 'velocity_x', 'velocity_y', 'velocity_magnitude'
- velocity_x: horizontal velocity (positive = right, negative = left)
- velocity_y: vertical velocity (positive = up, negative = down)
- Focus on using velocity to encourage better movement patterns
- The info dictionary also contains level metadata

VELOCITY-BASED IMPROVEMENTS:
```python
def reward_fn(state, action, info, original_reward=0):
    vel_x = info.get('velocity_x', 0)
    vel_y = info.get('velocity_y', 0)
    
    # Focus on steady rightward progress
    progress_reward = max(vel_x * 0.15, 0)
    
    # Discourage erratic movement
    stability_bonus = 0.1 if abs(vel_y) < 0.5 else 0
    
    return original_reward + progress_reward + stability_bonus
```

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
        "The info dictionary contains level metadata and velocity information (velocity_x, velocity_y, velocity_magnitude)."
    )

    user_msg = f"""
The current reward function passed preference evaluation (TPE score ≥ 0.8).

Current reward function:

{current_code}

Would you like to optionally improve or refine this function to further optimize performance or explore alternate behaviors?
Write a new version of `reward_fn(state, action, info, original_reward=0)` if you want.

IMPORTANT:
- Use original_reward parameter as the base reward signal
- The info dictionary contains velocity information: 'velocity_x', 'velocity_y', 'velocity_magnitude'
- velocity_x: horizontal velocity (positive = right, negative = left)
- velocity_y: vertical velocity (positive = up, negative = down)
- Focus on using velocity to encourage efficient movement patterns

VELOCITY OPTIMIZATION IDEAS:
```python
def reward_fn(state, action, info, original_reward=0):
    vel_x = info.get('velocity_x', 0)
    vel_y = info.get('velocity_y', 0)
    speed = info.get('velocity_magnitude', 0)
    
    # Encourage consistent rightward momentum
    momentum_reward = vel_x * 0.2 if vel_x > 0 else vel_x * 0.05
    
    # Reward smooth movement (low Y velocity variance)
    smoothness_bonus = 0.1 if abs(vel_y) < 1.0 else 0
    
    # Speed efficiency bonus
    efficiency_bonus = min(speed * 0.03, 0.15)
    
    return original_reward + momentum_reward + smoothness_bonus + efficiency_bonus
```

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