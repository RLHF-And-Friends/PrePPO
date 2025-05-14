from datasets import load_dataset, DatasetDict

from pre_ppo.utils import push_dataset_to_hub_with_retries
from pre_ppo.data_utils import cat_columns_contents


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

new_dataset = DatasetDict({
    "train": train_dataset,
    "validation": eval_dataset,
})

# Push dataset to hub
# =================================================================================================

NEW_DATASET_NAME = "tldr-preference"

push_dataset_to_hub_with_retries(
    new_dataset,
    repo_id=f"RLHF-And-Friends/{NEW_DATASET_NAME}",
)
