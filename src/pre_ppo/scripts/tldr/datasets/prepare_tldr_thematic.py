import typing as tp

import re
from tqdm import tqdm

from datasets import load_dataset, DatasetDict, Dataset

from pre_ppo.utils import push_dataset_to_hub_with_retries


# Load dataset
# =================================================================================================

dataset = load_dataset("trl-lib/tldr")

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]


# Collect all different subreddits
# =================================================================================================

def get_subreddit(prompt: str) -> str:
    match = re.search(r'^SUBREDDIT:\s*r/(\w+)', prompt)
    subreddit = match.group(1)

    return subreddit


def get_subreddits(dataset: Dataset) -> tp.Set[str]:
    subreddits = set()
    for sample in tqdm(dataset):
        prompt = sample["prompt"]
        subreddits.add(get_subreddit(prompt))

    return subreddits


train_subreddits = get_subreddits(train_dataset)
val_subreddits = get_subreddits(eval_dataset)
test_subreddits = get_subreddits(test_dataset)

print(
    f"Train subreddits:\n {train_subreddits}\n\n"
    f"Validation subreddits:\n {val_subreddits}\n\n"
    f"Test subreddits:\n {test_subreddits}"
)


# Make new dataset with subsets corresponding to subreddits
# =================================================================================================

def subreddit_filter(element, subreddit: str):
    prompt = element["prompt"]
    elem_subreddit = get_subreddit(prompt)
    
    return subreddit == elem_subreddit


subsets: tp.Dict[str, DatasetDict] = {}
for subreddit in train_subreddits:
    subreddit_subset_train = train_dataset.filter(
        subreddit_filter,
        fn_kwargs={"subreddit": subreddit}
    )
    subreddit_subset_val = eval_dataset.filter(
        subreddit_filter,
        fn_kwargs={"subreddit": subreddit}
    )
    subreddit_subset_test = test_dataset.filter(
        subreddit_filter,
        fn_kwargs={"subreddit": subreddit}
    )
    subreddit_subset = DatasetDict({
        "train": subreddit_subset_train,
        "validation": subreddit_subset_val,
        "test": subreddit_subset_test
    })
    subsets[subreddit] = subreddit_subset


# Loaad new dataset to hub
# =================================================================================================

repo_id = "anonymous-organization/tldr-thematic"

for subreddit, subset in subsets.items():
    subset.push_to_hub(repo_id, config_name=subreddit)
