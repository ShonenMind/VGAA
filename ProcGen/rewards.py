import os
import re
from openai import OpenAI
from datetime import datetime

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

import os
from datetime import datetime

def extract_code_block(response_text):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    lines = response_text.strip().splitlines()
    code_lines = [line for line in lines if line.strip() and not line.strip().lower().startswith("explanation")]
    if code_lines:
        return "\n".join(code_lines).strip()

    filename = f"logs/invalid_llm_response_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write("Full LLM response:\n\n")
        f.write(response_text.strip() + "\n")
    raise ValueError(f"No valid Python code block found in LLM response. Logged to {filename}")

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
        "agents in the Procgen CoinRun environment. Reward functions should guide the agent "
        "toward collecting coins and moving rightward (increasing x-position), while also allowing "
        "for novelty or exploration incentives. CoinRun rewards typically depend on fields in `info` "
        "(like 'x_pos', 'coins_collected', 'velocity')."
    )

    k_shot_example = """Here is an example of the format you should use:

```python
def reward_fn(state, action, info):
    x = info.get("x_pos", 0)
    coins = info.get("coins_collected", 0)
    return x * 0.01 + coins
```"""

    user_msg = f"""
Write a Python function named `reward_fn(state, action, info)` that returns a float reward.

Constraints:
- Use fields from `info` (like 'x_pos', 'velocity', 'coins_collected').
- Encourage progress (e.g. increasing x-position) and/or coin collection.
- Make the reward slightly unusual or interesting (e.g. shaping, small randomness, smooth changes), but avoid nonsensical or purely random formulas.
- Do NOT use external packages or unsafe code.
- The function must be valid, executable Python and must return a float.

ONLY output one Python code block like ```python ... ``` and nothing else.

{k_shot_example}
"""

    return _run_llm_prompt(system_msg, user_msg, temperature=0.9)


def get_reactive_reward_fn(previous_code: str, trajectory_summary: str):
    system_msg = "You are an assistant that writes Python reward functions for the Procgen CoinRun environment."

    user_msg = f"""
Your previous reward function did not work well and failed preference evaluation.

Previous reward function:

{previous_code}

Trajectory statistics:

{trajectory_summary}

Please fix the reward function. Write a new function named `reward_fn(state, action, info)` that returns a float.
Focus on improving agent performance.

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
    system_msg = "You are an assistant that helps improve Python reward functions for the Procgen CoinRun environment."

    user_msg = f"""
The current reward function passed preference evaluation (TPE score ≥ 0.8).

Current reward function:

{current_code}

Would you like to optionally improve or refine this function to further optimize performance or explore alternate behaviors?
Write a new version of `reward_fn(state, action, info)` if you want.

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