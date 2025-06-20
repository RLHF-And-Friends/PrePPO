import os

import copy

import torch

from datasets import load_dataset
from tokenizers import processors

from transformers import (
    AutoModelForCausalLM, AutoTokenizer
)

from peft import get_peft_model, TaskType, prepare_model_for_kbit_training

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    DataCollatorForCompletionOnlyLM
)

from pre_ppo.utils import push_to_hub_with_retries


###################################################################################################
# NAMES & PATHS
###################################################################################################

# Model path
# =================================================================================================
# MODEL_PATH = "meta-llama/Llama-3.2-1B"
MODEL_PATH = "Qwen/Qwen2-0.5B"
MODEL_NAME = MODEL_PATH.split('/')[1]

# Dataset path
# =================================================================================================
DATASET_PATH        = "RLHF-And-Friends/tldr-sft"
DATASET_TRAIN_SPLIT = "train"
DATASET_VAL_SPLIT   = "validation"
DATASET_NAME        = DATASET_PATH.split('/')[1]

# Project name
# =================================================================================================
PROJECT_NAME = "SFT-TLDR"
EXP_NAME = f"{MODEL_NAME}-SMALL"

# WandB
# =================================================================================================
os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_ENTITY"]  = "RADFAN"


###################################################################################################
# CONFIGS
###################################################################################################

TRAIN_SIZE  = 16722
EVAL_SIZE   = 1500

# Model config
# =================================================================================================

model_config = ModelConfig(
    model_name_or_path   = MODEL_PATH,

    # LoRA
    # ---------------------------------------------------------------------------------------------
    use_peft             = True,
    lora_task_type       = TaskType.CAUSAL_LM,
    use_rslora           = False,
    lora_r               = 16,
    lora_alpha           = 32,
    lora_dropout         = 0.0,
    lora_target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj"],
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

training_args = SFTConfig(
    # SFT trainer params
    # ---------------------------------------------------------------------------------------------

    dataset_num_proc            = 16,
    max_seq_length              = 1024,

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
    learning_rate = 1e-4,

    # Logs
    # ---------------------------------------------------------------------------------------------
    logging_steps               = 20,
    eval_strategy               = "steps",
    eval_steps                  = 200,

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

initial_tokenizer = AutoTokenizer.from_pretrained(
    model_config.model_name_or_path
)

tokenizer = AutoTokenizer.from_pretrained(
    model_config.model_name_or_path,
    pad_token = "<|pad|>",
    # add_eos_token = True, # To add EOS token during tokenization
    padding_side = "right",
)

# Add postrprocessor to add EOS token (Obligatory for Llama3 models only)
# -------------------------------------------------------------------------------------------------

# tokenizer._tokenizer.post_processor = processors.TemplateProcessing(
#     single=f"{tokenizer.bos_token}:0 $A:0 {tokenizer.eos_token}:0",
#     pair=f"{tokenizer.bos_token}:0 $A:0 {tokenizer.bos_token}:1 $B:1 {tokenizer.eos_token}:1",
#     special_tokens=[
#         (tokenizer.bos_token, tokenizer.bos_token_id),
#         (tokenizer.eos_token, tokenizer.eos_token_id),
#     ],
# )

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
model = AutoModelForCausalLM.from_pretrained(
    model_config.model_name_or_path,
    num_labels = 1,
    quantization_config = quantization_config,
    torch_dtype = torch_dtype,
)
initial_model_config = copy.deepcopy(model.config)

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


# Data collator to mask prompt labels
# =================================================================================================

# masking_collator = DataCollatorForCompletionOnlyLM(
#     # response_template=[11521, 28745, 4232, 28747], # Mistral
#     response_template="TL;DR:",
#     tokenizer = tokenizer
# )

###################################################################################################
# TRAINING
###################################################################################################

# Train
# =================================================================================================
def main() -> None:
    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        # data_collator    = masking_collator,
    )
    trainer.train()

    # Delete PAD token from the model's vocabulary
    # ---------------------------------------------------------------------------------------------
    trainer.model.resize_token_embeddings(initial_model_config.vocab_size)

    # Revert tokenizer to the initial state
    # ---------------------------------------------------------------------------------------------
    trainer.processing_class = initial_tokenizer

    # Merge LoRA adapters into the model
    # ---------------------------------------------------------------------------------------------
    trainer.model = trainer.model.merge_and_unload()
    
    # Set initial model config
    # ---------------------------------------------------------------------------------------------
    trainer.model.config = initial_model_config

    # Push model to hub with retries
    # ---------------------------------------------------------------------------------------------
    push_to_hub_with_retries(trainer, DATASET_NAME)


# Accelerate entry-point
# =================================================================================================
if __name__ == "__main__":
    main()
