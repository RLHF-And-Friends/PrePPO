from datasets import (
    load_dataset,
    get_dataset_config_names,
    DatasetDict,
)

DATASET_PATH = "GEM/wiki_lingua"

all_configs = get_dataset_config_names(DATASET_PATH)

configs = ['it', 'ko', 'es', 'tr', 'nl', 'vi', 'hi', 'fr', 'th', 'en', 'de', 
           'ar', 'pt', 'ru', 'ja', 'cs', 'zh', 'id']

subsets = {}
for config in configs:
    train_subset = load_dataset(
        DATASET_PATH,
        name=config,
        trust_remote_code=True,
        split="train"
    )
    train_subset = train_subset.remove_columns([
        "source_language", "target_language", "references"
    ])
    validation_subset = load_dataset(
        DATASET_PATH,
        name=config,
        trust_remote_code=True,
        split="validation"
    )
    validation_subset = validation_subset.remove_columns([
        "source_language", "target_language", "references"
    ])
    test_subset = load_dataset(
        DATASET_PATH,
        name=config,
        trust_remote_code=True,
        split="test"
    )
    test_subset = test_subset.remove_columns([
        "source_language", "target_language", "references"
    ])
    subset = DatasetDict({
        "train": train_subset,
        "validation": validation_subset,
        "test": test_subset,
    })
    subsets[config] = subset


repo_id = "RLHF-And-Friends/wiki_lingua"

for config, subset in subsets.items():
    subset.push_to_hub(repo_id, config_name=config)
