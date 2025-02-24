from datasets import load_dataset, Dataset

from hf_ppo.utils import get_responses
from hf_ppo.scripts.datasets.data_utils import push_to_hub_with_retries


# #################################################################################################
# NAMES & PATHS
# #################################################################################################

# Base model
# -------------------------------------------------------------------------------------------------
BASE_MODEL_PATH = "mistral-community/Mistral-7B-v0.2"
# -------------------------------------------------------------------------------------------------

# Fine-tuned model
# -------------------------------------------------------------------------------------------------
FT_MODEL_PATH = "RLHF-And-Friends/TLDR-Mistral-7B-SFT-PPO"
# -------------------------------------------------------------------------------------------------
FT_MODEL_NAME = FT_MODEL_PATH.split('/')[1]

# Dataset
# -------------------------------------------------------------------------------------------------
DATASET_PATH = "trl-lib/tldr"
DATASET_SPLIT = "test"
PROMPT_FIELD = "prompt"
SIZE = 100
# -------------------------------------------------------------------------------------------------

DATASET_NAME = DATASET_PATH.split('/')[1]


# #################################################################################################
# INFERENCE
# #################################################################################################

# Load prompt dataset
# =================================================================================================

test_dataset = load_dataset(DATASET_PATH, split=DATASET_SPLIT).select(range(SIZE))

prompts = list(test_dataset[PROMPT_FIELD])


# Inference
# =================================================================================================

base_completions = get_responses(prompts, BASE_MODEL_PATH, batch_size=32)
ft_completions = get_responses(prompts, FT_MODEL_PATH, batch_size=32)


# Save responses as dataset
# =================================================================================================

responses_dataset = Dataset.from_dict(
    {
        'prompt': prompts, 
        'base_completion': base_completions,
        'ft_completions': ft_completions
    }
)

push_to_hub_with_retries(
    responses_dataset,
    repo_id=f"RLHF-And-Friends/{DATASET_NAME}-{FT_MODEL_NAME}-completions",
    split = "test"
)
