from datasets import load_dataset, DatasetDict

from pre_ppo.utils import push_dataset_to_hub_with_retries


# Load dataset
# =================================================================================================

dataset = load_dataset("yahma/alpaca-cleaned")

dataset = dataset["train"].train_test_split(test_size=0.1)

dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# Push dataset to hub
# =================================================================================================

NEW_DATASET_NAME = "alpaca-cleaned"

push_dataset_to_hub_with_retries(
    dataset,
    repo_id=f"RLHF-And-Friends/{NEW_DATASET_NAME}",
)
