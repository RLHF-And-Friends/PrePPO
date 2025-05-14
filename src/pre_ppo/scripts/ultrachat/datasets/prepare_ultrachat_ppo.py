from datasets import load_dataset

from transformers import AutoTokenizer

from pre_ppo.utils import push_dataset_to_hub_with_retries
from pre_ppo.data_utils import apply_chat_template, DatasetFormat


# Load dataset
# =================================================================================================

dataset = load_dataset("HuggingFaceH4/ultrachat_200k")


# Drop 'prompt_id' and 'messages' columns from dataset
# =================================================================================================

dataset = dataset.remove_columns(["prompt_id", "messages"])


# Apply chat template
# =================================================================================================

TOKENIZER_PATH = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

dataset = dataset.map(
    apply_chat_template,
    fn_kwargs={
        "tokenizer": tokenizer,
        "columns_to_apply_to": ["prompt"],
        "dataset_format": DatasetFormat.PLAIN,
        "add_generation_prompt": True,
    },
    desc = "Applying chat template to the dataset..",
    load_from_cache_file=False,
)


# Push dataset to hub
# =================================================================================================

NEW_DATASET_NAME = "ultrachat-preprocessed"

push_dataset_to_hub_with_retries(
    dataset,
    repo_id=f"RLHF-And-Friends/{NEW_DATASET_NAME}",
)
