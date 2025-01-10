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
Run the script with
    accelerate launch path/to/this/script.py

To do so make sure that in advance you have
    - Set healthy device map
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    - Configured accelerate for multi-gpu setting (e.g via wizard)
    accelerate config

And that's it.
"""

### Models & Dataset

# Policy model path
# =================================================================================================
POLICY_PATH = "meta-llama/Llama-3.2-1B-Instruct"
# =================================================================================================
POLICY_NAME = POLICY_PATH.split('/')[1]

# Reward model path
# =================================================================================================
REWARD_PATH = "RLHF-And-Friends/Llama-3.2-1B-Instruct-Reward-ultrafeedback_binarized-LoRA-8r"
# =================================================================================================
REWARD_NAME = REWARD_PATH.split('/')[1]

# Prompts dataset path
# =================================================================================================
DATASET_PATH        = "HuggingFaceH4/ultrachat_200k"
DATASET_TRAIN_SPLIT = "train_gen"
DATASET_VAL_SPLIT   = "test_gen"
# =================================================================================================
DATASET_NAME        = DATASET_PATH.split('/')[1]

### WandB settings

os.environ["WANDB_PROJECT"] = f"Distributed-PPO"
os.environ["WANDB_ENTITY"] = "RADFAN"

### Models' configs

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

### PPO Trainer config

ppo_config = PPOConfig(
    # Common
    # ---------------------------------------------------------------------------------------------
    exp_name            = f"{POLICY_NAME}-Dual-GPU-BatchSize-16",
    output_dir          = f"{os.environ['WANDB_PROJECT']}-LoRA-{policy_model_config.lora_r}",
    dataset_num_proc    = 16,
    num_mini_batches    = 1,
    learning_rate       = 1e-5,
    # Make sure the desired effective batch size == batch_size * accum_steps * num_devices
    per_device_train_batch_size = 1,
    per_device_eval_batch_size  = 2,
    gradient_accumulation_steps = 1,
    num_train_epochs    = 1,
    response_length     = 512,
    stop_token          = "eos",
    # Logging
    # ---------------------------------------------------------------------------------------------
    save_steps          = 100,
    
    # Push to hub after training
    # ---------------------------------------------------------------------------------------------
    push_to_hub         = True,
    hub_model_id        = f"RLHF-And-Friends/{POLICY_NAME}-PPO-{DATASET_NAME}"
                          f"-LoRA-{policy_model_config.lora_r}",

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

### Initialize models and tokenizer

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

# SFT model
# -------------------------------------------------------------------------------------------------

sft_policy = AutoModelForCausalLM.from_pretrained(
    policy_model_config.model_name_or_path,
)
sft_policy.resize_token_embeddings(len(tokenizer), mean_resizing=False)
sft_policy.config.pad_token_id = tokenizer.pad_token_id

# Trainable policy
# -------------------------------------------------------------------------------------------------

policy = get_peft_model(sft_policy, get_peft_config(policy_model_config))

# Base model for Value and Reward models
# -------------------------------------------------------------------------------------------------

base_value_head_model = AutoModelForSequenceClassification.from_pretrained(
    policy_model_config.model_name_or_path,
    num_labels = 1,
)
base_value_head_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
base_value_head_model.config.pad_token_id = tokenizer.pad_token_id

# Value model with LoRA
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

### Initialize dataset

train_dataset = load_dataset(
    DATASET_PATH, 
    split=DATASET_TRAIN_SPLIT
).select(range(2000))
eval_dataset = load_dataset(
    DATASET_PATH, 
    split=DATASET_VAL_SPLIT
).select(range(100))

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

### Training

trainer = CustomPPOTrainer(
    args            = ppo_config,
    processing_class  = tokenizer,
    model             = policy,
    ref_model         = sft_policy,
    reward_model      = reward_model,
    value_model       = value_model,
    train_dataset     = train_dataset,
    eval_dataset      = eval_dataset,
)

trainer.train()

### Save model

# Remove added pad token from model's embedding layer

policy.resize_token_embeddings(len(tokenizer) - 1)

# trainer.push_to_hub(dataset_name=DATASET_PATH) not for multi-GPU
