from enum import Enum

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

# Append TL;DR function
# -------------------------------------------------------------------------------------------------

def append_tldr(element, column: str):
    element[column] += "\n\nTL;DR: "

    return element
