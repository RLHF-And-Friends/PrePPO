from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
)

from peft import PeftModelForSequenceClassification

from trl import ModelConfig, get_quantization_config

from fed_ppo.utils import apply_chat_template, tokenize, DatasetFormat


###################################################################################################
# NAMES & PATHS
###################################################################################################

# Base model path
# =================================================================================================
BASE_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"

# Reward adapter path
# =================================================================================================
REWARD_ADAPTER_PATH = "RLHF-And-Friends/Llama-3.2-1B-Instruct-Reward-Ultrafeedback-QLoRA-Normalized"

# Normalization dataset
# =================================================================================================
NORM_DATASET_PATH  = "HuggingFaceH4/ultrachat_200k"
NORM_DATASET_SPLIT = "train_sft"

# Normalized adapter path to save
# =================================================================================================
NORM_ADAPTER_PATH = "RLHF-And-Friends/Llama-3.2-1B-Instruct-Reward-Ultrafeedback-QLoRA-Normalized"

###################################################################################################
# CONFIGS
###################################################################################################

model_config = ModelConfig(
    torch_dtype               = "bfloat16",
    load_in_8bit              = False,
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    use_bnb_nested_quant      = True,
    
)

###################################################################################################
# TOKENIZER & MODELS
###################################################################################################

# Tokenizer
# =================================================================================================

tokenizer = AutoTokenizer.from_pretrained(
    REWARD_ADAPTER_PATH, 
    use_fast=True,
)

# Model
# =================================================================================================

device = torch.device("cuda:0")

quantization_config = get_quantization_config(model_config)

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH, 
    num_labels = 1,
    quantization_config = quantization_config,
    device_map = "auto",
    torch_dtype = model_config.torch_dtype
)

model = PeftModelForSequenceClassification.from_pretrained(
    base_model,
    REWARD_ADAPTER_PATH
)

model.to(device)
model.eval()

# Sync padding tokens
# =================================================================================================

model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model.config.pad_token_id = tokenizer.pad_token_id

print(model)

###################################################################################################
# DATASET
###################################################################################################

# Load dataset
# =================================================================================================

dataset = load_dataset(
    NORM_DATASET_PATH
)[NORM_DATASET_SPLIT]

dataset = dataset.select(range(100))

# Apply chat template
# =================================================================================================

dataset = dataset.map(
    apply_chat_template,
    fn_kwargs = {
        "tokenizer": tokenizer,
        "columns_to_apply_to": ["messages"],
        "dataset_format": DatasetFormat.CONVERSATIONAL,
        "add_generation_prompt": False,
        "new_columns": ["chat"],
    },
    batched = True
)


# Tokenize
# =================================================================================================

dataset = dataset.map(
    tokenize,
    fn_kwargs = {
        "tokenizer": tokenizer,
        "columns_to_apply_to": ["chat"],
        "columns_for_ids": ["input_ids"],
        "columns_for_attn": ["attention_mask"]
    },
    batched=True
)


# Create dataloader
# =================================================================================================

dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=DataCollatorWithPadding(tokenizer),
    shuffle=False
)

###################################################################################################
# Normalization
###################################################################################################

reward_sum = 0
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Get model predictions
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        reward_sum += torch.mean(logits).item()

mean_reward = reward_sum / len(dataloader)

print(f"Mean reward: {mean_reward}")


###################################################################################################
# Add bias to the head and save
###################################################################################################

# Add bias
# =================================================================================================

# Original head
original_head = model.score.modules_to_save.default

# New head
new_layer = torch.nn.Linear(in_features=2048, out_features=1, bias=True)

# Copy weights
new_layer.weight.data = original_head.weight.data.clone()

new_layer.bias.data = torch.tensor(-mean_reward)

model.score.modules_to_save.default = new_layer

# Remove pad token and save
# =================================================================================================

print(model.score.modules_to_save.default.bias)

model.resize_token_embeddings(len(tokenizer) - 1)

model.push_to_hub(NORM_ADAPTER_PATH)
tokenizer.push_to_hub(NORM_ADAPTER_PATH)
