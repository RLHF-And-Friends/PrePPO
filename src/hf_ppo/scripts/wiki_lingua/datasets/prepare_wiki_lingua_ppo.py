from tqdm import tqdm

from datasets import load_dataset, Dataset, get_dataset_split_names

from hf_ppo.data_utils import append_tldr


# Paths and variables
# =================================================================================================

# Initial dataset
# -------------------------------------------------------------------------------------------------
DATASET_PATH = "RLHF-And-Friends/wiki_lingua"
CONFIGURATIONS = ["en", "ru", "de", "fr", "es", "it", "nl"]

TEXT_FIELD = "source"
SUMMARY_FIELD = "target"

# Target dataset
# -------------------------------------------------------------------------------------------------
TARGET_DATASET_PATH = "RLHF-And-Friends/wiki_lingua_ppo"

# Generation
# =================================================================================================

for configuration in tqdm(CONFIGURATIONS, desc="Processing configurations"):

    dataset = load_dataset(DATASET_PATH, configuration)

    dataset_split_names = get_dataset_split_names(DATASET_PATH, configuration)

    for split in dataset_split_names:
        split_dataset = dataset[split]
        
        split_dataset = split_dataset.map(append_tldr, fn_kwargs={"column": TEXT_FIELD})
        
        split_texts = split_dataset[TEXT_FIELD]
        split_summaries = split_dataset[SUMMARY_FIELD]

        new_dataset_split = Dataset.from_dict({
            "text": split_texts,
            "summary": split_summaries,
        })

        new_dataset_split.push_to_hub(
            repo_id=TARGET_DATASET_PATH,
            config_name=configuration,
            split=split
        )
