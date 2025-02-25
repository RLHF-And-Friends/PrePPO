from datasets import load_dataset

from trl import OpenAIPairwiseJudge

from huggingface_hub import HfApi

'''
set API key in advance:

export OPENAI_API_KEY=<API key here>
'''

# #################################################################################################
# NAMES & PATHS
# #################################################################################################

DATASET_PATH = "RLHF-And-Friends/tldr-TLDR-Mistral-7B-SFT-PPO-completions"
NUM_SAMPLES = 100


# #################################################################################################
# Evaluation
# #################################################################################################

# System prompt
# =================================================================================================

SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''

# Load responses
# =================================================================================================

responses_dataset = load_dataset(DATASET_PATH)["test"].select(range(NUM_SAMPLES))

prompts = list(responses_dataset['prompt'])
completions = [
    [base_completion, ft_completion] 
    for base_completion, ft_completion in zip(
        responses_dataset['base_completion'], responses_dataset['ft_completions']
    )
]

# Judge
# =================================================================================================

MODEL_TO_USE = "gpt-4o-mini"

gpt_judge = OpenAIPairwiseJudge(
    model = MODEL_TO_USE,
    system_prompt=SYSTEM_PROMPT
)

gpt_judgements = gpt_judge.judge(prompts, completions, shuffle_order=False)

gpt_winrate = sum(gpt_judgements) / len(gpt_judgements)

print(f"GPT-judged winrate: {gpt_winrate}")


# Add winrate metric to README.md
# =================================================================================================

readme_text = f"""---
language: en
metrics:
  - GPT-based winrate: {gpt_winrate}
---

Winrate based on {MODEL_TO_USE} opinion: {gpt_winrate}.
"""

api = HfApi()
api.upload_file(
    path_or_fileobj=readme_text.encode(),  # Convert string to bytes
    path_in_repo="README.md",
    repo_id=DATASET_PATH,
    repo_type="dataset",
    commit_message="Add evaluation metric to README"
)
