import os

import torch
from torch.optim import AdamW

from datasets import load_dataset

from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer
)

from peft import get_peft_model, TaskType, prepare_model_for_kbit_training

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    get_peft_config,
    get_quantization_config,
)

from fed_ppo.utils import (
    custom_optimizer, 
    apply_chat_template,
    tokenize,
    OptimizerConfig,
)

###################################################################################################
# NAMES AND PATHS
###################################################################################################

# Model path
# =================================================================================================
MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME = MODEL_PATH.split('/')[1]

# Dataset path
# =================================================================================================
DATASET_PATH        = "trl-lib/ultrafeedback_binarized"
DATASET_TRAIN_SPLIT = "train"
DATASET_VAL_SPLIT   = "test"
DATASET_NAME        = DATASET_PATH.split('/')[1]

# Project name
# =================================================================================================
PROJECT_NAME = f"Reward-Modelling-{MODEL_NAME}-{DATASET_NAME}"
EXP_NAME = f"LoRA-8r"

# WandB
# =================================================================================================
os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_ENTITY"]  = "RADFAN"


###################################################################################################
# CONFIGS
###################################################################################################

# Datasets will be filtered according to max length
MAX_LENGTH = 1024

# Model config
# =================================================================================================

model_config = ModelConfig(
    model_name_or_path   = MODEL_PATH,
    # LoRA
    # ---------------------------------------------------------------------------------------------
    use_peft             = True,
    lora_task_type       = TaskType.SEQ_CLS,
    use_rslora           = False,
    lora_r               = 8,
    lora_alpha           = 16,
    lora_dropout         = 0.0,
    lora_target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj"],
    # Head will require grad automatically
    lora_modules_to_save = None,
    # Quantization
    # ---------------------------------------------------------------------------------------------
    load_in_8bit         = False,
    load_in_4bit         = False,
    bnb_4bit_quant_type  = "nf4",
    use_bnb_nested_quant = True,
    torch_dtype          = "bfloat16",
)

# Reward trainer config
# =================================================================================================

training_args = RewardConfig(
    # Reward trainer params
    # ---------------------------------------------------------------------------------------------
    max_length                  = None,
    dataset_num_proc            = 16,
    center_rewards_coefficient  = None,
    # Common
    # ---------------------------------------------------------------------------------------------
    run_name                    = EXP_NAME,
    output_dir                  = f"~/hf/models/{PROJECT_NAME}/{EXP_NAME}",
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    num_train_epochs            = 2,
    gradient_checkpointing      = False,
    gradient_accumulation_steps = 4,
    bf16                        = True,

    # Frequency of logs
    # ---------------------------------------------------------------------------------------------
    logging_steps               = 20,

    # Evaluation
    # ---------------------------------------------------------------------------------------------
    eval_strategy               = "steps",
    eval_steps                  = 100,

    # Push to hub after training
    # ---------------------------------------------------------------------------------------------
    push_to_hub                 = False,
    hub_model_id                = f"RLHF-And-Friends/{EXP_NAME}",
)

# Optimizer config
# =================================================================================================

optimizer_config = OptimizerConfig(
    optimizer_type = AdamW,
    layer_lr       = {
        "lora":  1e-5, # LoRA adapters
        "score": 1e-4, # Head
    }
)


###################################################################################################
# TOKENIZER & MODELS
###################################################################################################

# Model
# =================================================================================================

# Make quantization config
# -------------------------------------------------------------------------------------------------
quantization_config = get_quantization_config(model_config)

# Use KV-cache or not
# -------------------------------------------------------------------------------------------------
use_cache = False if training_args.gradient_checkpointing else True

# Set model type
# -------------------------------------------------------------------------------------------------

if model_config.torch_dtype is not None:
    torch_dtype = getattr(torch, model_config.torch_dtype)

# Create model
# -------------------------------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_config.model_name_or_path, 
    num_labels = 1,
    quantization_config = quantization_config,
    device_map = "auto",
    use_cache = use_cache,
    trust_remote_code = True,
    torch_dtype = torch_dtype
)
if model_config.load_in_4bit or model_config.load_in_8bit:
    model = prepare_model_for_kbit_training(
        model,
        training_args.gradient_checkpointing,
    )

# Wrap in LoRA
# -------------------------------------------------------------------------------------------------
lora_config = get_peft_config(model_config)
model = get_peft_model(model, lora_config)

# Tokenizer
# =================================================================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_config.model_name_or_path, 
    use_fast=True
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# Sync padding tokens
# =================================================================================================

model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model.config.pad_token_id = tokenizer.pad_token_id


###################################################################################################
# DATASET
###################################################################################################

# Model
# =================================================================================================

# Make quantization config
# -------------------------------------------------------------------------------------------------
quantization_config = get_quantization_config(model_config)

# Use KV-cache or not
# -------------------------------------------------------------------------------------------------
use_cache = False if training_args.gradient_checkpointing else True

# Set model type
# -------------------------------------------------------------------------------------------------

if model_config.torch_dtype is not None:
    torch_dtype = getattr(torch, model_config.torch_dtype)

# Create model
# -------------------------------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_config.model_name_or_path, 
    num_labels = 1,
    quantization_config = quantization_config,
    device_map = "auto",
    use_cache = use_cache,
    trust_remote_code = True,
    torch_dtype = torch_dtype
)
if model_config.load_in_4bit or model_config.load_in_8bit:
    model = prepare_model_for_kbit_training(
        model,
        training_args.gradient_checkpointing,
    )

# Wrap in LoRA
# -------------------------------------------------------------------------------------------------
lora_config = get_peft_config(model_config)
model = get_peft_model(model, lora_config)

# Tokenizer
# =================================================================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_config.model_name_or_path, 
    use_fast=True
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# Sync padding tokens
# =================================================================================================

model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model.config.pad_token_id = tokenizer.pad_token_id


###################################################################################################
# DATASET
###################################################################################################

dataset = load_dataset(DATASET_PATH)

train_dataset = dataset[DATASET_TRAIN_SPLIT]
eval_dataset = dataset[DATASET_VAL_SPLIT]

# Apply chat tamplate and tokenize beforehand to avoid doing it inside 
# the 'RewardTrainer'
# -------------------------------------------------------------------------------------------------

train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    load_from_cache_file=False
)
eval_dataset = eval_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    load_from_cache_file = False
)

train_dataset = train_dataset.map(
    tokenize,
    fn_kwargs={"tokenizer": tokenizer},
    load_from_cache_file = False
)
eval_dataset = eval_dataset.map(
    tokenize,
    fn_kwargs={"tokenizer": tokenizer},
    load_from_cache_file = False
)

# Filter datasets by length (keep only examples which are no longer then 
# `max_length` tokens)
# -------------------------------------------------------------------------------------------------

length_filter = (
    lambda x: len(x["input_ids_chosen"]) <= MAX_LENGTH
              and len(x["input_ids_rejected"]) <= MAX_LENGTH
)

train_dataset = train_dataset.filter(
    length_filter,
    num_proc=training_args.dataset_num_proc,
)

eval_dataset = eval_dataset.filter(
    length_filter,
    num_proc=training_args.dataset_num_proc,
)


###################################################################################################
# TRAINING
###################################################################################################

optimizer = custom_optimizer(model, optimizer_config)

trainer = RewardTrainer(
    model            = model,
    processing_class = tokenizer,
    args             = training_args,
    train_dataset    = train_dataset,
    eval_dataset     = eval_dataset,
    optimizers       = (optimizer, None)
)

trainer.train()

# Remove added pad token from model's embedding layer
model.resize_token_embeddings(len(tokenizer) - 1)

trainer.push_to_hub(dataset_name=DATASET_PATH)
