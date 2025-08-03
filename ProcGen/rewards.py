import os
import re
from openai import OpenAI


openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def extract_code_block(response_text):
   """
   Extracts Python code from a markdown-style code block (```python ... ```).
   Raises an error if no valid code block is found.
   """
   match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
   if not match:
       raise ValueError("No valid Python code block found in LLM response.")
   return match.group(1).strip()


def get_random_reward_fn():
   system_msg = (
       "You are an assistant that writes Python reward functions for reinforcement learning "
       "agents in the Procgen CoinRun environment."
   )


   user_msg = """
Write a Python function named `reward_fn(state, action, info)` that returns a float reward.
You do NOT know exact details, but have a vague idea that CoinRun involves collecting coins and moving forward.
Make this reward function somewhat random or unusual but valid Python.
Return a reward based on 'info' or 'state' that might encourage some novel behavior.
Do NOT use external APIs or unsafe code.
Only output a single Python code block like ```python ... ```. Do not include any explanations or text outside the code block.
"""


   response = client.chat.completions.create(
       model="gpt-4",
       messages=[
           {"role": "system", "content": system_msg},
           {"role": "user", "content": user_msg},
       ],
       temperature=1.0,
       max_tokens=300,
   )


   full_response = response.choices[0].message.content.strip()
   code = extract_code_block(full_response)
   return code


def get_refined_reward_fn(previous_code: str, trajectory_summary: str):
   system_msg = (
       "You are an assistant that writes Python reward functions for the Procgen CoinRun environment."
   )


   user_msg = f"""
Previous reward function code:


{previous_code}


Agent performed with these trajectory stats:


{trajectory_summary}


Please provide a new improved Python reward function named `reward_fn(state, action, info)` to improve agent behavior.
Only output a single Python code block like ```python ... ```. Do not include any explanations or text outside the code block.
"""


   response = client.chat.completions.create(
       model="gpt-4",
       messages=[
           {"role": "system", "content": system_msg},
           {"role": "user", "content": user_msg},
       ],
       temperature=0.7,
       max_tokens=300,
   )


   full_response = response.choices[0].message.content.strip()
   code = extract_code_block(full_response)
   return code


if __name__ == "__main__":
   code = get_random_reward_fn()
   print("=== Random initial reward function ===\n")
   print(code)