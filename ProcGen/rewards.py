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


def _run_llm_prompt(system_msg, user_msg, temperature=0.7, max_tokens=800):
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
    import numpy as np
    
    # Extract observation for velocity info
    obs = state.get('obs', None)
    
    # Shape the original reward from the environment
    shaped_reward = original_reward * 2.0
    
    # Add exploration bonus
    exploration_bonus = 0.01
    
    # Extract velocity from painted velocity boxes (top-left corner)
    if obs is not None and obs.shape == (64, 64, 3):
        # Get velocity from the two 8x8 painted squares in top-left corner
        # Left box (horizontal velocity): obs[0:8, 0:8] - white=rightward, black=leftward
        # Right box (vertical velocity): obs[0:8, 8:16] - white=upward, black=downward
        
        # Sample center pixels of each box to get velocity values
        h_vel_pixel = obs[4, 4]  # Center of left box (horizontal velocity)
        v_vel_pixel = obs[4, 12]  # Center of right box (vertical velocity)
        
        # Decode velocity: grayscale value indicates velocity direction/magnitude
        # White (255) = max positive velocity, Black (0) = max negative velocity, Gray (128) = zero
        h_velocity = (np.mean(h_vel_pixel) - 128) / 128.0  # Normalize to -1 to +1
        v_velocity = (np.mean(v_vel_pixel) - 128) / 128.0
        
        # Reward rightward movement and jumping, penalize going backwards
        velocity_bonus = h_velocity * 0.1 + max(0, v_velocity) * 0.05
    else:
        velocity_bonus = 0
    
    # Add some small randomness for diversity
    import random
    noise = random.uniform(-0.005, 0.005)
    
    return shaped_reward + exploration_bonus + velocity_bonus + noise
```"""

    user_msg = f"""
Write a Python function named `reward_fn(state, action, info, original_reward=0)` that returns a float reward.

IMPORTANT CONSTRAINTS:
- The info dictionary ONLY contains: 'prev_level_seed', 'prev_level_complete', 'level_seed', 'TimeLimit.truncated'
- Game state (position, coins) is NOT available in info - it's encoded in the 64x64x3 pixel observation
- Use the original_reward parameter as your base - it contains the environment's built-in reward signal
- You can shape/scale the original reward and add bonuses for exploration, action diversity, etc.
- Do NOT try to access non-existent keys like 'x_pos', 'coins_collected', 'velocity'

VELOCITY INFORMATION AVAILABLE (PAINTED VELOCITY BOXES):
- ProcGen CoinRun paints velocity information as TWO colored squares in the top-left corner
- Each square is approximately 8x8 pixels in size (for 64x64 observations)
- LEFT BOX (pixels [0:8, 0:8]): Horizontal velocity
  * White pixels (high values ~255) = rightward movement
  * Black pixels (low values ~0) = leftward movement  
  * Gray pixels (~128) = no horizontal movement
- RIGHT BOX (pixels [0:8, 8:16]): Vertical velocity
  * White pixels (high values ~255) = upward movement (jumping)
  * Black pixels (low values ~0) = downward movement (falling)
  * Gray pixels (~128) = no vertical movement
- To extract: sample center pixels like obs[4, 4] for horizontal, obs[4, 12] for vertical
- Decode velocity: (pixel_value - 128) / 128.0 gives normalized velocity in range [-1, +1]
- Use this for sophisticated movement-based reward shaping (e.g., reward rightward progress, jumping)

Focus on:
- Scaling/shaping the original_reward
- Adding small exploration bonuses  
- Encouraging action diversity
- Using velocity information from painted pixels to reward good movement
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

VELOCITY INFORMATION AVAILABLE (PAINTED VELOCITY BOXES):
- ProcGen CoinRun paints velocity information as TWO colored squares in the top-left corner
- Each square is approximately 8x8 pixels in size (for 64x64 observations)
- LEFT BOX (pixels [0:8, 0:8]): Horizontal velocity
  * White pixels (high values ~255) = rightward movement
  * Black pixels (low values ~0) = leftward movement  
  * Gray pixels (~128) = no horizontal movement
- RIGHT BOX (pixels [0:8, 8:16]): Vertical velocity
  * White pixels (high values ~255) = upward movement (jumping)
  * Black pixels (low values ~0) = downward movement (falling)
  * Gray pixels (~128) = no vertical movement
- To extract: sample center pixels like obs[4, 4] for horizontal, obs[4, 12] for vertical
- Decode velocity: (pixel_value - 128) / 128.0 gives normalized velocity in range [-1, +1]
- Use this for sophisticated movement-based reward shaping (e.g., reward rightward progress, jumping)

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

VELOCITY INFORMATION AVAILABLE (PAINTED VELOCITY BOXES):
- ProcGen CoinRun paints velocity information as TWO colored squares in the top-left corner
- Each square is approximately 8x8 pixels in size (for 64x64 observations)
- LEFT BOX (pixels [0:8, 0:8]): Horizontal velocity
  * White pixels (high values ~255) = rightward movement
  * Black pixels (low values ~0) = leftward movement  
  * Gray pixels (~128) = no horizontal movement
- RIGHT BOX (pixels [0:8, 8:16]): Vertical velocity
  * White pixels (high values ~255) = upward movement (jumping)
  * Black pixels (low values ~0) = downward movement (falling)
  * Gray pixels (~128) = no vertical movement
- To extract: sample center pixels like obs[4, 4] for horizontal, obs[4, 12] for vertical
- Decode velocity: (pixel_value - 128) / 128.0 gives normalized velocity in range [-1, +1]
- Use this for sophisticated movement-based reward shaping (e.g., reward rightward progress, jumping)

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