from datasets import load_dataset, Dataset

from huggingface_hub import HfApi

from pre_ppo.utils import get_responses, push_dataset_to_hub_with_retries


# #################################################################################################
# NAMES & PATHS
# #################################################################################################

# LHS model
# -------------------------------------------------------------------------------------------------
LHS_MODEL_PATH = "RLHF-And-Friends/TLDR-Llama-3.1-8B-Base-PPO"
LHS_TOKENIZER_PATH = None
# -------------------------------------------------------------------------------------------------
LHS_MODEL_NAME = LHS_MODEL_PATH.split('/')[1]

# RHS model
# -------------------------------------------------------------------------------------------------
RHS_MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
RHS_TOKENIZER_PATH = None
# -------------------------------------------------------------------------------------------------
RHS_MODEL_NAME = RHS_MODEL_PATH.split('/')[1]

# Dataset
# -------------------------------------------------------------------------------------------------
DATASET_PATH = "RLHF-And-Friends/tldr-ppo"
DATASET_SPLIT = "test"
PROMPT_FIELD = "prompt"
SIZE = 100
# -------------------------------------------------------------------------------------------------
DATASET_NAME = DATASET_PATH.split('/')[1]

# HF repo
# -------------------------------------------------------------------------------------------------
HF_REPO_ID = "RLHF-And-Friends/Llama-Base-TLDR-PPO-vs-Llama-Base"

README_TEXT = f"""---
tags: [rlhf, tldr, radfan]
---

This dataset contains responses of two models given prompt.

The column "prompt" contains prompt given to both models. Two other columns contain reponses of respective models.

Models used:

Left: **{LHS_MODEL_NAME}**,
Right: **{RHS_MODEL_NAME}**.

Original dataset with prompts: **{DATASET_PATH}**.
"""

# #################################################################################################
# INFERENCE
# #################################################################################################

# Load prompt dataset
# =================================================================================================

test_dataset = load_dataset(DATASET_PATH, split=DATASET_SPLIT).select(range(SIZE))

prompts = list(test_dataset[PROMPT_FIELD])


# Inference
# =================================================================================================

lhs_completions = get_responses(
    prompts=prompts,
    model_path=LHS_MODEL_PATH,
    tokenizer_path=LHS_TOKENIZER_PATH,
    batch_size=32
)
rhs_completions = get_responses(
    prompts=prompts,
    model_path=RHS_MODEL_PATH,
    tokenizer_path=RHS_TOKENIZER_PATH,
    batch_size=32
)


# Save responses as dataset
# =================================================================================================

# Create dataset and push it to HF Hub
# -------------------------------------------------------------------------------------------------

responses_dataset = Dataset.from_dict(
    {
        'prompt': prompts, 
        f'{LHS_MODEL_NAME}': lhs_completions,
        f'{RHS_MODEL_NAME}': rhs_completions
    },
    split="test"
)

push_dataset_to_hub_with_retries(
    responses_dataset,
    repo_id=f"{HF_REPO_ID}",
)

# Add README.md for a pretty dataset card
# -------------------------------------------------------------------------------------------------

api = HfApi()
api.upload_file(
    path_or_fileobj=README_TEXT.encode(),  # Convert string to bytes
    path_in_repo="README.md",
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    commit_message="Add README.md"
)
