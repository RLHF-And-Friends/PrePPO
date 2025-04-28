import re
from tqdm import tqdm

from datasets import load_dataset, DatasetDict

from hf_ppo.utils import push_dataset_to_hub_with_retries


# Load dataset
# =================================================================================================

dataset = load_dataset("trl-lib/tldr")

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]

subreddits = set()
for sample in tqdm(train_dataset):
    prompt = sample["prompt"]
    subreddit = re.search(r'^SUBREDDIT:\s*r/(\w+)', prompt).group(1)
    subreddits.add(subreddit)

print(subreddits)
