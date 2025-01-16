import os
import torch

from datasets import load_dataset

from transformers import (
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding
)

from peft import (
    PeftModelForSequenceClassification,
    TaskType,
    get_peft_model,
    # prepare_model_for_kbit_training,
)

from trl import (
    ModelConfig,
    PPOConfig,
    get_peft_config,
    get_quantization_config,
)

from fed_ppo.ppo_trainer import CustomPPOTrainer
from fed_ppo.utils import (
    apply_chat_template,
    tokenize,
    DatasetFormat,
    set_bias
)

from fed_ppo.prompts import (
    STAY_WITHIN_THE_TOKEN_LIMIT,
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

###################################################################################################
# NAMES AND PATHS
###################################################################################################

# Policy model path
# =================================================================================================
POLICY_PATH = "meta-llama/Llama-3.2-1B-Instruct"
POLICY_NAME = POLICY_PATH.split('/')[1]

# Reward model path
# =================================================================================================
REWARD_PATH = "RLHF-And-Friends/Llama-3.2-1B-Instruct-Reward-Ultrafeedback-QLoRA-Normalized"
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
EXP_NAME = f"{POLICY_NAME}-4xA4000-NormRM-Q4-LoRA-8-Batch-48-TokIO-960-512"

# WandB
# =================================================================================================
os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_ENTITY"] = "RADFAN"

# PyTorch
# =================================================================================================
# alleviates cuda memory segmenation and allows to train with larger effecitve
# batch sizes with no eventual CUDA out of memory errors
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

###################################################################################################
# Configs
###################################################################################################

TOK_IN_MAX = 960
TOK_OUT_MAX = 512

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
    torch_dtype          = "bfloat16",
    load_in_8bit         = False,
    load_in_4bit         = True,
    bnb_4bit_quant_type  = "nf4",
    use_bnb_nested_quant = True,
)

# Value mode4
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
    torch_dtype         = "bfloat16",
    load_in_8bit        = False,
    load_in_4bit        = True,
    bnb_4bit_quant_type  = "nf4",
    use_bnb_nested_quant = True,
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
    exp_name            = EXP_NAME,
    output_dir          = f"~/hf/models/{PROJECT_NAME}/{EXP_NAME}",
    dataset_num_proc    = 16,
    # ppo epoch size = data batch size = effective ppo batch size * num mini batches
    # effecitve ppo batch size = per_dev_tr_bs * grad_acc_st * num_dev
    num_ppo_epochs                  = 1, # num ppo epochs per rollout batch
    num_mini_batches                = 3, # batches in ppo epoch
    per_device_train_batch_size     = 1, # \
                                         #  > effective local ppo batch size
    gradient_accumulation_steps     = 4, # /
    local_rollout_forward_batch_size= 64,# response generation and processing batch size
    # per_device_eval_batch_size  = 1,
    num_train_epochs    = 1,
    response_length     = TOK_OUT_MAX,
    stop_token          = "eos",
    # Logging
    # ---------------------------------------------------------------------------------------------
    save_steps          = 480,
    # Push to hub after training
    # ---------------------------------------------------------------------------------------------
    push_to_hub         = True,
    hub_model_id        = f"RLHF-And-Friends/{EXP_NAME}",
    # Optimizer params
    # ---------------------------------------------------------------------------------------------
    learning_rate       = 1e-5,
    adam_epsilon        = 1e-5,
    # On-policy params
    # ---------------------------------------------------------------------------------------------
    missing_eos_penalty = 1.0,  # as in N+ PPO implementation details
    # PPO params
    # ---------------------------------------------------------------------------------------------
    whiten_rewards      = False, # normalize reward in batch
    kl_coef             = 0.05,
    cliprange           = 0.2,
    vf_coef             = 0.1,
    cliprange_value     = 0.2,
    gamma               = 1.0,
    lam                 = 0.95,
)


###################################################################################################
# TOKENIZER & MODELS
###################################################################################################

# Tokenizer
# =================================================================================================
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    policy_model_config.model_name_or_path,
    padding_side = "left",
    pad_token = "<|pad|>",
)

# Models
# =================================================================================================

# SFT
# -------------------------------------------------------------------------------------------------
sft_policy = LlamaForCausalLM.from_pretrained(
    policy_model_config.model_name_or_path,
    quantization_config=get_quantization_config(policy_model_config),
    torch_dtype=getattr(torch, policy_model_config.torch_dtype)
)
sft_policy.resize_token_embeddings(len(tokenizer), mean_resizing=False)
sft_policy.config.pad_token_id = tokenizer.pad_token_id

# Policy
# -------------------------------------------------------------------------------------------------
policy = get_peft_model(sft_policy, get_peft_config(policy_model_config))

# Shared model for Value and Reward
# -------------------------------------------------------------------------------------------------
base_value_head_model = LlamaForSequenceClassification.from_pretrained(
    policy_model_config.model_name_or_path,
    num_labels=1,
    quantization_config=get_quantization_config(value_model_config),
    torch_dtype=getattr(torch, value_model_config.torch_dtype)
)
# Add bias to the model head since LlamaForSequenceCLassification does not do it
set_bias(
    base_value_head_model,
    "score",
    bias=0.0,
    dtype=getattr(torch, value_model_config.torch_dtype)
)

base_value_head_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
base_value_head_model.config.pad_token_id = tokenizer.pad_token_id

# Reward model
# -------------------------------------------------------------------------------------------------
# Load LoRA adapters for reward model
reward_model = PeftModelForSequenceClassification.from_pretrained(
    base_value_head_model,
    reward_model_config.model_name_or_path,
)

# Value model
# -------------------------------------------------------------------------------------------------
# Create value model from reward model as recommended in N+ PPO impl. details
value_model = PeftModelForSequenceClassification.from_pretrained(
    base_value_head_model,
    reward_model_config.model_name_or_path,
    is_trainable = True
)


###################################################################################################
# DATASET
###################################################################################################

# Load dataset
# =================================================================================================

train_dataset = load_dataset(
    DATASET_PATH,
    split=DATASET_TRAIN_SPLIT
).select(range(48000))

eval_dataset = load_dataset(
    DATASET_PATH,
    split=DATASET_VAL_SPLIT
).select(range(480))

# Apply chat template
# =================================================================================================

train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={
        "tokenizer": tokenizer,
        "columns_to_apply_to": ["prompt"],
        "dataset_format": DatasetFormat.PLAIN,
        "add_generation_prompt": True,
        "system_prompt": STAY_WITHIN_THE_TOKEN_LIMIT(TOK_OUT_MAX)
    },
)
eval_dataset = eval_dataset.map(
    apply_chat_template,
    fn_kwargs={
        "tokenizer": tokenizer,
        "columns_to_apply_to": ["prompt"],
        "dataset_format": DatasetFormat.PLAIN,
        "add_generation_prompt": True,
        "system_prompt": STAY_WITHIN_THE_TOKEN_LIMIT(TOK_OUT_MAX)
    },
)

# Tokenize
# =================================================================================================

train_dataset = train_dataset.map(
    tokenize,
    fn_kwargs={
        "tokenizer": tokenizer,
        "columns_to_apply_to": ["prompt"],
        "columns_for_ids": ["input_ids"],
        "columns_for_attn": ["attention_mask"] # not really used here
    },
    batched = True
)
eval_dataset = eval_dataset.map(
    tokenize,
    fn_kwargs={
        "tokenizer": tokenizer,
        "columns_to_apply_to": ["prompt"],
        "columns_for_ids": ["input_ids"],
        "columns_for_attn": ["attention_mask"] # not really used here
    },
    batched = True
)

# Filter prompts by length
# =================================================================================================

def len_filter(x) -> bool:
    return len(x["input_ids"]) <= TOK_IN_MAX

train_dataset = train_dataset.filter(
    len_filter,
    num_proc=ppo_config.dataset_num_proc,
)

eval_dataset = eval_dataset.filter(
    len_filter,
    num_proc=ppo_config.dataset_num_proc
)

# Create data collator with padding
# =================================================================================================

data_collator = DataCollatorWithPadding(
    tokenizer = tokenizer,
    padding = "max_length",
    max_length = TOK_IN_MAX,
)

# Remove unnecessary columns
# =================================================================================================

train_dataset = train_dataset.remove_columns(["prompt", "prompt_id", "messages"])
eval_dataset = eval_dataset.remove_columns(["prompt", "prompt_id", "messages"])


###################################################################################################
# TRAINING
###################################################################################################

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
        data_collator     = data_collator,
    )
    trainer.train()


# Accelerate entry-point
# =================================================================================================
if __name__ == "__main__":
    main()
