from datasets import (
    load_dataset,
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names
)

from hf_ppo.utils import push_dataset_to_hub_with_retries
from hf_ppo.data_utils import cat_columns_contents, append_tldr


# Load dataset
# =================================================================================================

dataset_path = "RLHF-And-Friends/wiki_lingua_preference"
all_configs = get_dataset_config_names(dataset_path)

train_subsets = []
validation_subsets = []
for config in all_configs:
    # Train subset
    # ---------------------------------------------------------------------------------------------
    train_subset = load_dataset(dataset_path, name=config, split="train")
    train_subset = train_subset.add_column(
        name="language",
        column=[config]*len(train_subset)
    )
    train_subset = train_subset.map(append_tldr, fn_kwargs={"column": "text"})
    train_subsets.append(train_subset)

    # Validation subset
    # ---------------------------------------------------------------------------------------------
    validation_subset = load_dataset(dataset_path, name=config, split="validation")
    validation_subset = validation_subset.add_column(
        name="language",
        column=[config]*len(validation_subset)
    )
    validation_subset = validation_subset.map(append_tldr, fn_kwargs={"column": "text"})
    validation_subsets.append(validation_subset)

train_dataset = concatenate_datasets(train_subsets)
validation_dataset = concatenate_datasets(validation_subsets)

# Cat prompt and completions
# =================================================================================================

train_dataset = train_dataset.map(
    cat_columns_contents,
    fn_kwargs={
        "lhs_column_names": ["text", "text"],
        "rhs_column_names": ["chosen", "rejected"],
        "cat_column_names": ["chosen", "rejected"],
    },
    desc = "Train dataset concatenation",
    load_from_cache_file=False,
)

validation_dataset = validation_dataset.map(
    cat_columns_contents,
    fn_kwargs={
        "lhs_column_names": ["text", "text"],
        "rhs_column_names": ["chosen", "rejected"],
        "cat_column_names": ["chosen", "rejected"],
    },
    desc = "Eval dataset concatenation",
    load_from_cache_file=False,
)

new_dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
})

# Push dataset to hub
# =================================================================================================

NEW_DATASET_NAME = "wiki-lingua-reward"

push_dataset_to_hub_with_retries(
    new_dataset,
    repo_id=f"RLHF-And-Friends/{NEW_DATASET_NAME}",
)
