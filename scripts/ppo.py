import os

from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from peft import (
    PeftModelForSequenceClassification,
    TaskType, 
    get_peft_model
)

from trl import (
    ModelConfig,
    PPOConfig,
    get_peft_config,
)

from fed_ppo.ppo_trainer import CustomPPOTrainer
from fed_ppo.utils import apply_chat_template, tokenize
from fed_ppo.prompts import (
    STAY_WITHIN_THE_TOKEN_LIMIT, 
    STAY_WITHIN_THE_TOKEN_LIMIT_TRAININIG_AWARE
)


"""
Run this script with single-GPU normally with:
    >>> (env) user@machine:~$ python path/to/this/script.py

Or run the script in distributed setup with:
    >>> (env) user@machine:~$ accelerate launch path/to/this/script.py

In this case make sure that in advance you have
    - Set healthy device map
    >>> (env) user@machine:~$ export CUDA_DEVICE_ORDER=PCI_BUS_ID

    - Configured accelerate for multi-gpu setting (e.g via wizard)
    >>> (env) user@machine:~$ accelerate config

    - Set proper GPUs at the previous step or with CUDA_VISIBLE_DEVICES (in
    this case leave GPU ids field to 'all')

And that's it.
"""


# NAMES AND PATHS


# Policy model path
# =================================================================================================
POLICY_PATH = "meta-llama/Llama-3.2-1B-Instruct"
POLICY_NAME = POLICY_PATH.split('/')[1]

# Reward model path
# =================================================================================================
REWARD_PATH = "RLHF-And-Friends/Llama-3.2-1B-Instruct-Reward-Ultrafeedback"
REWARD_NAME = REWARD_PATH.split('/')[1]

# Dataset path
# =================================================================================================
DATASET_PATH        = "HuggingFaceH4/ultrachat_200k"
DATASET_TRAIN_SPLIT = "train_gen"
DATASET_VAL_SPLIT   = "test_gen"
DATASET_NAME        = DATASET_PATH.split('/')[1]

# Project name
# =================================================================================================
PROJECT_NAME = "Distributed-PPO"
EXP_NAME = f"{POLICY_NAME}-Dual-GPU-BatchSize-16"

### WandB
# =================================================================================================
os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_ENTITY"] = "RADFAN"


### Configs

# Policy
# =================================================================================================
policy_model_config = ModelConfig(
    model_name_or_path   = POLICY_PATH,
    # LoRA
    # ---------------------------------------------------------------------------------------------
    use_peft             = True,
    lora_r               = 8,
    lora_alpha           = 16,
    lora_dropout         = 0.0,
    lora_task_type       = TaskType.CAUSAL_LM,
    lora_target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj"],
    # Quantization
    # ---------------------------------------------------------------------------------------------
    load_in_8bit         = False,
    load_in_4bit         = False,
    torch_dtype          = "bfloat16",
)

# Value model
# =================================================================================================
value_model_config = ModelConfig(
    # LoRA
    # ---------------------------------------------------------------------------------------------
    use_peft            = True,
    lora_r              = 8,
    lora_alpha          = 16,
    lora_dropout        = 0.0,
    lora_task_type      = TaskType.SEQ_CLS,
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    # Quantization
    # ---------------------------------------------------------------------------------------------
    load_in_8bit        = False,
    load_in_4bit        = False,
    torch_dtype         = "bfloat16",
)

# Reward model
# =================================================================================================
reward_model_config = ModelConfig(
    model_name_or_path  = REWARD_PATH,
    load_in_8bit        = False,
    load_in_4bit        = False,
)

### PPO Trainer
ppo_config = PPOConfig(
    # Common
    # ---------------------------------------------------------------------------------------------
    exp_name            = EXP_NAME
    output_dir          = f"~/hf/models/{PROJECT_NAME}/{EXP_NAME}",
    dataset_num_proc    = 16,
    num_mini_batches    = 1,
    learning_rate       = 1e-5,
    # Make sure the desired effective batch size == batch_size * accum_steps * num_devices
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 2,
    num_train_epochs    = 1,
    response_length     = 512,
    stop_token          = "eos",
    # Logging
    # ---------------------------------------------------------------------------------------------
    save_steps          = 160, 
    # Push to hub after training
    # ---------------------------------------------------------------------------------------------
    push_to_hub         = True,
    hub_model_id        = f"RLHF-And-Friends/{EXP_NAME}",
    # On-policy params
    # ---------------------------------------------------------------------------------------------
    missing_eos_penalty = 0.0,
    local_rollout_forward_batch_size = 1,
    # PPO params
    # ---------------------------------------------------------------------------------------------
    num_ppo_epochs      = 1,
    whiten_rewards      = False,
    kl_coef             = 0.05,
    cliprange           = 0.2,
    vf_coef             = 0.1,
    cliprange_value     = 0.2,
    gamma               = 1.0,
    lam                 = 0.95,
)


# Tokenizer
# =================================================================================================
tokenizer = AutoTokenizer.from_pretrained(
    policy_model_config.model_name_or_path,
    use_fast     = True,
    padding_side = "left"
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


# Models
# =================================================================================================

# SFT
# -------------------------------------------------------------------------------------------------
sft_policy = AutoModelForCausalLM.from_pretrained(
    policy_model_config.model_name_or_path,
)
sft_policy.resize_token_embeddings(len(tokenizer), mean_resizing=False)
sft_policy.config.pad_token_id = tokenizer.pad_token_id

# Policy
# -------------------------------------------------------------------------------------------------
policy = get_peft_model(sft_policy, get_peft_config(policy_model_config))

# Shared model for Value and Reward
# -------------------------------------------------------------------------------------------------
base_value_head_model = AutoModelForSequenceClassification.from_pretrained(
    policy_model_config.model_name_or_path,
    num_labels = 1,
)
base_value_head_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
base_value_head_model.config.pad_token_id = tokenizer.pad_token_id

# Value model
# -------------------------------------------------------------------------------------------------
value_model = get_peft_model(
    base_value_head_model,
    get_peft_config(value_model_config)
)

# Reward model
# -------------------------------------------------------------------------------------------------
reward_model = PeftModelForSequenceClassification.from_pretrained(
    base_value_head_model,
    reward_model_config.model_name_or_path,
)

# Data
# =================================================================================================
train_dataset = load_dataset(
    DATASET_PATH, 
    split=DATASET_TRAIN_SPLIT
).select(range(16000))
eval_dataset = load_dataset(
    DATASET_PATH, 
    split=DATASET_VAL_SPLIT
).select(range(160))

train_dataset = train_dataset.remove_columns("messages")
eval_dataset = eval_dataset.remove_columns("messages")

train_dataset = train_dataset.map(
    apply_chat_template, 
    fn_kwargs={
        "tokenizer": tokenizer, 
        "system_prompt": STAY_WITHIN_THE_TOKEN_LIMIT(512)
    }, 
    load_from_cache_file=False
)
eval_dataset = eval_dataset.map(
    apply_chat_template, 
    fn_kwargs={
        "tokenizer": tokenizer,
    },
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

train_dataset = train_dataset.remove_columns(["prompt", "prompt_id"])
eval_dataset = eval_dataset.remove_columns(["prompt", "prompt_id"])


# Train
# =================================================================================================
def main() -> None:
    trainer = CustomPPOTrainer(
        args              = ppo_config,
        processing_class  = tokenizer,
        model             = policy,
        ref_model         = sft_policy,
        reward_model      = reward_model,
        value_model       = value_model,
        train_dataset     = train_dataset,
        eval_dataset      = eval_dataset,
    )
    trainer.train()


# Accelerate entry-point
# =================================================================================================
if __name__ == "__main__":
    main()

# Quite hard to do as the model is saved in inside the trainer
# Remove added pad token from model's embedding layer
# policy.resize_token_embeddings(len(tokenizer) - 1)
