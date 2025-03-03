import os

import torch

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

from hf_ppo.utils import (
    push_to_hub_with_retries
)

# #################################################################################################
# NAMES & PATHS
# #################################################################################################

# Model path
# =================================================================================================
MODEL_PATH = "RLHF-And-Friends/TLDR-Mistral-7B-SmallSFT"
MODEL_NAME = MODEL_PATH.split('/')[1]

# Dataset path
# =================================================================================================
DATASET_PATH        = "RLHF-And-Friends/tldr-preference"
DATASET_TRAIN_SPLIT = "train"
DATASET_VAL_SPLIT   = "validation"
DATASET_NAME        = DATASET_PATH.split('/')[1]

# Project name
# =================================================================================================
PROJECT_NAME = "RM-TLDR"
EXP_NAME = f"{MODEL_NAME}"

# WandB
# =================================================================================================
os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_ENTITY"]  = "RADFAN"


# #################################################################################################
# CONFIGS
# #################################################################################################

TRAIN_SIZE  = 92858
EVAL_SIZE   = 2000

# Model config
# =================================================================================================

model_config = ModelConfig(
    model_name_or_path   = MODEL_PATH,

    # LoRA
    # ---------------------------------------------------------------------------------------------
    use_peft             = True,
    lora_task_type       = TaskType.SEQ_CLS,
    use_rslora           = False,
    lora_r               = 16,
    lora_alpha           = 32,
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
    dataset_num_proc            = 16,
    center_rewards_coefficient  = None,
    max_length                  = 1024,

    # Common
    # ---------------------------------------------------------------------------------------------
    run_name                    = EXP_NAME,
    output_dir                  = f"~/hf/models/{PROJECT_NAME}/{EXP_NAME}",
    num_train_epochs            = 1,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size  = 2,
    gradient_accumulation_steps = 8,
    gradient_checkpointing      = False,
    bf16                        = True,
    
    # Optimizer
    # ---------------------------------------------------------------------------------------------
    learning_rate = 3e-6,

    # Logs
    # ---------------------------------------------------------------------------------------------
    logging_steps               = 20,
    eval_strategy               = "steps",
    eval_steps                  = 200,

    # Push to hub after training
    # ---------------------------------------------------------------------------------------------
    push_to_hub                 = False, # push manually with pad embedding removed
    hub_model_id                = f"RLHF-And-Friends/{PROJECT_NAME}-{EXP_NAME}",
)

# #################################################################################################
# TOKENIZER & MODELS
# #################################################################################################

# Tokenizer
# =================================================================================================

initial_tokenizer = AutoTokenizer.from_pretrained(
    model_config.model_name_or_path
)

tokenizer = AutoTokenizer.from_pretrained(
    model_config.model_name_or_path,
    pad_token = "<pad>",
    add_eos_token = True, # To add EOS token during tokenization
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
model = AutoModelForSequenceClassification.from_pretrained(
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


# #################################################################################################
# LOAD DATASET
# #################################################################################################

dataset = load_dataset(DATASET_PATH)

train_dataset = dataset["train"].select(range(TRAIN_SIZE))
eval_dataset = dataset["validation"].select(range(EVAL_SIZE))


# #################################################################################################
# TRAINING
# #################################################################################################

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

    # Delete PAD token from the model's vocabulary
    # ---------------------------------------------------------------------------------------------
    trainer.model.resize_token_embeddings(len(tokenizer) - 1)

    # Revert tokenizer to the initial state
    # ---------------------------------------------------------------------------------------------
    trainer.tokenizer = initial_tokenizer

    # Merge LoRA adapters into the model
    # ---------------------------------------------------------------------------------------------
    trainer.model = trainer.model.merge_and_unload()

    # Push model to hub
    # ---------------------------------------------------------------------------------------------
    push_to_hub_with_retries(trainer, DATASET_NAME)


# Accelerate entry-point
# =================================================================================================
if __name__ == "__main__":
    main()
