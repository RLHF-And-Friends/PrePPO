from enum import Enum

import time

from datasets import Dataset

from transformers import PreTrainedTokenizer


class DatasetFormat(str, Enum):
    PLAIN = "plain",
    CONVERSATIONAL = "conversational"


# Functions to use inside 'map' methhod
# =================================================================================================

# Apply chat template
# -------------------------------------------------------------------------------------------------

def apply_chat_template(
    element,
    tokenizer: PreTrainedTokenizer,
    columns_to_apply_to: list[str],
    dataset_format: DatasetFormat,
    add_generation_prompt: bool = False,
    new_columns: list[str] | None = None,
    system_prompt: str | None = None
):
    if new_columns is None:
        new_columns = columns_to_apply_to
    
    for column_name, new_column_name in zip(columns_to_apply_to, new_columns):
        if dataset_format is DatasetFormat.CONVERSATIONAL:
            prompt = element[column_name]
        elif dataset_format is DatasetFormat.PLAIN:
            if system_prompt is not None:
                prompt = [
                    {'role': "system", 'content': system_prompt},
                    {"role": "user", "content": element[column_name]}
                ]
            else:
                prompt = [
                    {'role': "user", "content": element[column_name]}
                ]
            
        element[new_column_name] = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )

    return element

# Tokenize
# -------------------------------------------------------------------------------------------------

def tokenize(
    element, 
    tokenizer: PreTrainedTokenizer,
    columns_to_apply_to: list[str],
    columns_for_ids: list[str],
    columns_for_attn: list[str],
    add_special_tokens = True,
):
    for column_name, ids_column_name, attn_column_name in zip(
        columns_to_apply_to, columns_for_ids, columns_for_attn
    ):
        tokenized = tokenizer(
            element[column_name], add_special_tokens=add_special_tokens
        )
        element[ids_column_name] = tokenized["input_ids"]
        element[attn_column_name] = tokenized["attention_mask"]

    return element

# Cat column contents
# -------------------------------------------------------------------------------------------------

def cat_columns_contents(
    element,
    lhs_column_names: list[str],
    rhs_column_names: list[str],
    cat_column_names: list[str]
):
    for lhs_column_name, rhs_column_name, cat_column_name in zip(
        lhs_column_names, rhs_column_names, cat_column_names
    ):
        element[cat_column_name] = element[lhs_column_name] + element[rhs_column_name]
        
    return element


# Other functions
# =================================================================================================

def push_to_hub_with_retries(
    dataset: Dataset,
    repo_id: str,
    split: str,
    dataset_card: str = "",
    max_retries: int = 10,
    delay: int = 5
):
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to push...")
            dataset.push_to_hub(repo_id=repo_id, split=split, )
            print("✅ Push successful!")
            return
        except Exception as e:
            print(f"❌ Push failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
            else:
                print("❌ Maximum retries reached. Push failed.")
