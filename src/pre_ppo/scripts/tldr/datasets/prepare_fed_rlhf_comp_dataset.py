from datasets import load_dataset

from huggingface_hub import HfApi

from pre_ppo.utils import get_responses, push_dataset_to_hub_with_retries
from pre_ppo.data_utils import cat_columns_contents


# #################################################################################################
# NAMES & PATHS
# #################################################################################################

# Model
# -------------------------------------------------------------------------------------------------
MODEL_PATH = "RLHF-And-Friends/TLDR-Llama-3.1-8B-SmallSFT-PPO"
TOKENIZER_PATH = None
# -------------------------------------------------------------------------------------------------
MODEL_NAME = MODEL_PATH.split('/')[1]

# Dataset
# -------------------------------------------------------------------------------------------------
DATASET_PATH = "RLHF-And-Friends/tldr-sft"
DATASET_SPLIT = "test"
PROMPT_FIELD = "prompt"
SIZE = 1000
# -------------------------------------------------------------------------------------------------
DATASET_NAME = DATASET_PATH.split('/')[1]

# HF repo
# -------------------------------------------------------------------------------------------------
HF_REPO_ID = "RLHF-And-Friends/Humans-vs-Llama-SmallSFT-PPO"

README_TEXT = f"""---
tags: [rlhf, tldr, radfan]
---

This dataset contains human completions from {DATASET_PATH} {DATASET_SPLIT} split and {MODEL_PATH} completions.

The column "prompt" contains prompt given both to humans and the models.

Model used:

Model: **{MODEL_NAME}**.

Original dataset with prompts and human completions: **{DATASET_PATH}**.
"""

# #################################################################################################
# FORMAT COMPLETIONS & INFERENCE
# #################################################################################################

# Load dataset with prompts and human completions and concat them
# =================================================================================================

test_dataset = load_dataset(DATASET_PATH, split=DATASET_SPLIT).select(range(SIZE))

test_dataset = test_dataset.map(
    cat_columns_contents,
    fn_kwargs={
        "lhs_column_names": ["prompt"],
        "rhs_column_names": ["completion"],
        "cat_column_names": ["human"],
    },
    desc = "Prompt completion concatenation",
    load_from_cache_file=False,
)
test_dataset = test_dataset.remove_columns(["completion"])

prompts = list(test_dataset[PROMPT_FIELD])

# Inference
# =================================================================================================

model_completions = get_responses(
    prompts=prompts,
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    batch_size=32
)


# Save responses as dataset
# =================================================================================================

# Create dataset and push it to HF Hub
# -------------------------------------------------------------------------------------------------

test_dataset = test_dataset.add_column(f"{MODEL_NAME}", model_completions)

push_dataset_to_hub_with_retries(
    test_dataset,
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
