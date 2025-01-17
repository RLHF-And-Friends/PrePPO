import os

import torch

from datasets import load_dataset

from transformers import (
    LlamaForSequenceClassification, PreTrainedTokenizerFast
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
    apply_chat_template,
    tokenize,
    DatasetFormat,
)

###################################################################################################
# NAMES & PATHS
###################################################################################################

# Model path
# =================================================================================================
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = MODEL_PATH.split('/')[1]

# Dataset path
# =================================================================================================
DATASET_PATH        = "trl-lib/ultrafeedback_binarized"
DATASET_TRAIN_SPLIT = "train"
DATASET_VAL_SPLIT   = "test"
DATASET_NAME        = DATASET_PATH.split('/')[1]

# Project name
# =================================================================================================
PROJECT_NAME = "RM-UltrafeedbackBinarized"
EXP_NAME = f"{MODEL_NAME}-Q4-LoRA8-Batch-16-Tok-1024"

# WandB
# =================================================================================================
os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_ENTITY"]  = "RADFAN"


###################################################################################################
# CONFIGS
###################################################################################################

# Datasets will be filtered according to max length
TOK_LIM     = 1024
TRAIN_SIZE  = 62100 # 62.100 MAX
EVAL_SIZE   = 1000  #  1.000 MAX

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
    load_in_4bit         = True,
    bnb_4bit_quant_type  = "nf4",
    use_bnb_nested_quant = True,
    torch_dtype          = "bfloat16",
)

# Reward trainer config
# =================================================================================================

training_args = RewardConfig(
    # Reward trainer params
    # ---------------------------------------------------------------------------------------------
    dataset_num_proc            = 16,
    center_rewards_coefficient  = None,

    # Common
    # ---------------------------------------------------------------------------------------------
    run_name                    = EXP_NAME,
    output_dir                  = f"~/hf/models/{PROJECT_NAME}/{EXP_NAME}",
    num_train_epochs            = 2,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 4,
    gradient_checkpointing      = False,
    bf16                        = True,

    # Optimizer
    # ---------------------------------------------------------------------------------------------
    learning_rate               = 1e-5,

    # Logs
    # ---------------------------------------------------------------------------------------------
    logging_steps               = 20,
    eval_strategy               = "steps",
    eval_steps                  = 500,

    # Push to hub after training
    # ---------------------------------------------------------------------------------------------
    push_to_hub                 = False, # would push manually with pad embedding removed
    hub_model_id                = f"RLHF-And-Friends/{PROJECT_NAME}-{EXP_NAME}",
)


###################################################################################################
# TOKENIZER & MODELS
###################################################################################################

# Tokenizer
# =================================================================================================

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    model_config.model_name_or_path,
    pad_token = "<|pad|>",
)

# Model
# =================================================================================================

# Make quantization config
# -------------------------------------------------------------------------------------------------
quantization_config = get_quantization_config(model_config)

# Set model type
# -------------------------------------------------------------------------------------------------

torch_dtype = getattr(torch, model_config.torch_dtype)

# Create model
# -------------------------------------------------------------------------------------------------
model = LlamaForSequenceClassification.from_pretrained(
    model_config.model_name_or_path,
    num_labels = 1,
    quantization_config = quantization_config,
    torch_dtype = torch_dtype,
)

if model_config.load_in_4bit or model_config.load_in_8bit:
    model = prepare_model_for_kbit_training(
        model,
        training_args.gradient_checkpointing,
    )

# Add padding token
# -------------------------------------------------------------------------------------------------

model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model.config.pad_token_id = tokenizer.pad_token_id

# Wrap in LoRA
# -------------------------------------------------------------------------------------------------
lora_config = get_peft_config(model_config)
model = get_peft_model(model, lora_config)


###################################################################################################
# DATASET
###################################################################################################

# Load dataset
# =================================================================================================
dataset = load_dataset(DATASET_PATH)

train_dataset = dataset[DATASET_TRAIN_SPLIT].select(range(TRAIN_SIZE))
eval_dataset = dataset[DATASET_VAL_SPLIT].select(range(EVAL_SIZE))

# Apply chat tamplate
# =================================================================================================

train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={
        "tokenizer": tokenizer,
        "columns_to_apply_to": ["chosen", "rejected"],
        "dataset_format": DatasetFormat.CONVERSATIONAL,
        "add_generation_prompt": False,
    },
    batched = True
)
eval_dataset = eval_dataset.map(
    apply_chat_template,
    fn_kwargs={
        "tokenizer": tokenizer,
        "columns_to_apply_to": ["chosen", "rejected"],
        "dataset_format": DatasetFormat.CONVERSATIONAL,
        "add_generation_prompt": False,
    },
    batched = True
)

# Tokenize
# =================================================================================================

train_dataset = train_dataset.map(
    tokenize,
    fn_kwargs={
        "tokenizer": tokenizer,
        "columns_to_apply_to": ["chosen", "rejected"],
        "columns_for_ids": ["input_ids_chosen", "input_ids_rejected"],
        "columns_for_attn": ["attention_mask_chosen", "attention_mask_rejected"],
    },
    batched = True
)
eval_dataset = eval_dataset.map(
    tokenize,
    fn_kwargs={
        "tokenizer": tokenizer,
        "columns_to_apply_to": ["chosen", "rejected"],
        "columns_for_ids": ["input_ids_chosen", "input_ids_rejected"],
        "columns_for_attn": ["attention_mask_chosen", "attention_mask_rejected"],
    },
    batched = True
)

# Filter datasets by length (keep only examples which are no longer then
# `max_length` tokens)
# =================================================================================================

def len_filter(x) -> bool:
    return len(x["input_ids_chosen"]) <= TOK_LIM and len(x["input_ids_rejected"]) <= TOK_LIM

train_dataset = train_dataset.filter(
    len_filter,
    num_proc=training_args.dataset_num_proc,
)

eval_dataset = eval_dataset.filter(
    len_filter,
    num_proc=training_args.dataset_num_proc,
)


###################################################################################################
# TRAINING
###################################################################################################

# Train
# =================================================================================================
def main() -> None:
    trainer = RewardTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
    )
    trainer.train()

    model.resize_token_embeddings(len(tokenizer) - 1)
    trainer.push_to_hub(dataset_name=DATASET_PATH)


# Accelerate entry-point
# =================================================================================================
if __name__ == "__main__":
    main()
