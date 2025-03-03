from datasets import load_dataset

from hf_ppo.utils import push_dataset_to_hub_with_retries
from hf_ppo.scripts.datasets.data_utils import cat_columns_contents


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

# Push dataset to hub
# =================================================================================================

NEW_DATASET_NAME = "tldr-ppo"

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

push_dataset_to_hub_with_retries(
    test_dataset,
    repo_id=f"RLHF-And-Friends/{NEW_DATASET_NAME}",
    split="test"
)
