from datasets import load_dataset, DatasetDict

from hf_ppo.utils import push_dataset_to_hub_with_retries


# Load dataset
# =================================================================================================

TRAIN_START = 20000
TRAIN_STOP = 116722

dataset = load_dataset("trl-lib/tldr")

train_dataset = dataset["train"].select(range(TRAIN_START, TRAIN_STOP))
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Drop 'completion' column from dataset
# =================================================================================================

train_dataset = train_dataset.remove_columns("completion")
eval_dataset = eval_dataset.remove_columns("completion")
test_dataset = test_dataset.remove_columns("completion")

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
