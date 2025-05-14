from datasets import load_dataset, DatasetDict

from pre_ppo.utils import push_dataset_to_hub_with_retries


# Load dataset
# =================================================================================================

TRAIN_START = 16722
TRAIN_STOP = 116722

dataset = load_dataset("trl-lib/tldr")

train_dataset = dataset["train"].select(range(TRAIN_START, TRAIN_STOP))
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]

new_dataset = DatasetDict({
    "train": train_dataset,
    "validation": eval_dataset,
    "test": test_dataset
})

new_dataset = DatasetDict({
    "train": train_dataset,
    "validation": eval_dataset,
    "test": test_dataset
})

# Push dataset to hub
# =================================================================================================

NEW_DATASET_NAME = "tldr-ppo"

push_dataset_to_hub_with_retries(
    new_dataset,
    repo_id=f"RLHF-And-Friends/{NEW_DATASET_NAME}",
)
