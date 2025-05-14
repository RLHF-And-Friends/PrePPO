import typing as tp

from datasets import load_dataset, DatasetDict, Dataset


# Load dataset
# =================================================================================================

dataset = load_dataset("nvidia/HelpSteer3", name="preference")

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]


# Leave only multilingual domain
# =================================================================================================

code_train_dataset = train_dataset.filter(
    lambda row: row["domain"] == "multilingual"
)
code_validation_dataset = validation_dataset.filter(
    lambda row: row["domain"] == "multilingual"
)


# Check what languages are available
# =================================================================================================

def get_available_languages(dataset: Dataset) -> tp.Set[str]:
    available_languages = set()
    for row in dataset:
        available_languages.add(row["language"])
        
    return available_languages

train_languages = get_available_languages(code_train_dataset)
validation_languages = get_available_languages(code_validation_dataset)

print(
    f"Train languages:\n {train_languages}\n\n"
    f"Validation languages:\n {validation_languages}"
)


# Make new dataset with the subset for each language
# =================================================================================================

def choose_best_response(sample):
    if sample["overall_preference"] <= 0:
        sample["best_response"] = sample["response1"]
    else:
        sample["best_response"] = sample["response2"]

    return sample


subsets: tp.Dict[str, DatasetDict] = {}
for language in train_languages:
    subset_train = code_train_dataset.filter(
        lambda row: row["language"] == language
    )
    subset_train = subset_train.map(
        choose_best_response
    )

    subset_validation = code_validation_dataset.filter(
        lambda row: row["language"] == language
    )
    subset_validation = subset_validation.map(
        choose_best_response
    )

    subset = DatasetDict({
        "train": subset_train,
        "validation": subset_validation,
    })
    subsets[language] = subset


# Loaad new dataset to hub
# =================================================================================================

repo_id = "RLHF-And-Friends/helpsteer3-multilingual"

for language, subset in subsets.items():
    subset.push_to_hub(repo_id, config_name=language)
