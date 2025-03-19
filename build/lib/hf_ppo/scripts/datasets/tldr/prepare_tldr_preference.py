from datasets import load_dataset

from hf_ppo.utils import push_dataset_to_hub_with_retries
from hf_ppo.data_utils import cat_columns_contents


# Load dataset
# =================================================================================================

dataset = load_dataset("trl-lib/tldr-preference")

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# Cat prompt and completions
# =================================================================================================

train_dataset = train_dataset.map(
    cat_columns_contents,
    fn_kwargs={
        "lhs_column_names": ["prompt", "prompt"],
        "rhs_column_names": ["chosen", "rejected"],
        "cat_column_names": ["chosen", "rejected"],
    },
    desc = "Train dataset concatenation",
    load_from_cache_file=False,
)

eval_dataset = eval_dataset.map(
    cat_columns_contents,
    fn_kwargs={
        "lhs_column_names": ["prompt", "prompt"],
        "rhs_column_names": ["chosen", "rejected"],
        "cat_column_names": ["chosen", "rejected"],
    },
    desc = "Eval dataset concatenation",
    load_from_cache_file=False,
)

# Push dataset to hub
# =================================================================================================

NEW_DATASET_NAME = "tldr-preference"

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
