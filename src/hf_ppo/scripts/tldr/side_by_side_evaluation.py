from datasets import load_dataset

from trl import OpenAIPairwiseJudge


'''
set API key in advance:

export OPENAI_API_KEY=<API key here>
'''

# #################################################################################################
# NAMES & PATHS
# #################################################################################################

DATASET_PATH = "RLHF-And-Friends/tldr-TLDR-Mistral-7B-SFT-PPO-completions"
SPLIT = "test"
PROMPT_COLUMN = "prompt"
NUM_SAMPLES_TO_USE = 100


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

responses_dataset = load_dataset(DATASET_PATH)[SPLIT].select(range(NUM_SAMPLES_TO_USE))

column_names = responses_dataset.column_names
completion_columns = [
    column_name for column_name in column_names if column_name != PROMPT_COLUMN
]
assert len(completion_columns) == 2

prompts = list(responses_dataset[PROMPT_COLUMN])
completions = [
    [lhs, rhs]
    for lhs, rhs in zip(
        responses_dataset[completion_columns[0]], responses_dataset[completion_columns[1]]
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
