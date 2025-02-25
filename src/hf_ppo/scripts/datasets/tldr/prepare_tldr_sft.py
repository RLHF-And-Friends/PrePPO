from datasets import load_dataset

from hf_ppo.utils import push_dataset_to_hub_with_retries
from hf_ppo.scripts.datasets.data_utils import cat_columns_contents


# Load dataset
# =================================================================================================

TRAIN_SIZE = 20000
EVAL_SIZE  = 2000

dataset = load_dataset("trl-lib/tldr")

train_dataset = dataset["train"].select(range(TRAIN_SIZE))
eval_dataset = dataset["validation"].select(range(EVAL_SIZE))

# Push dataset to hub
# =================================================================================================

NEW_DATASET_NAME = "tldr-sft"

push_dataset_to_hub_with_retries(
    train_dataset,
    repo_id=f"RLHF-And-Friends/{NEW_DATASET_NAME}",
    split="train"
)

push_dataset_to_hub_with_retries(
    eval_dataset,
    repo_id=f"RLHF-And-Friends/{NEW_DATASET_NAME}",
    split="validation"
)
