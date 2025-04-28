from datasets import load_dataset, DatasetDict

from hf_ppo.utils import push_dataset_to_hub_with_retries


# Load dataset
# =================================================================================================

TRAIN_SIZE = 16722
EVAL_SIZE  = 2000

dataset = load_dataset("trl-lib/tldr")

train_dataset = dataset["train"].select(range(TRAIN_SIZE))
eval_dataset = dataset["validation"].select(range(EVAL_SIZE))
test_dataset = dataset["test"]

new_dataset = DatasetDict({
    "train": train_dataset,
    "validation": eval_dataset,
    "test": test_dataset
})

# Push dataset to hub
# =================================================================================================

NEW_DATASET_NAME = "tldr-sft"

push_dataset_to_hub_with_retries(
    new_dataset,
    repo_id=f"RLHF-And-Friends/{NEW_DATASET_NAME}",
)
