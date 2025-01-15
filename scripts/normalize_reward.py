from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trl import ModelConfig, get_quantization_config

###################################################################################################
# NAMES & PATHS
###################################################################################################

# Base model path
# =================================================================================================
BASE_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"

# Reward adapter path
# =================================================================================================
REWARD_ADAPTER_PATH = "RLHF-And-Friends/Llama-3.2-1B-Instruct-Reward-Ultrafeedback-QLoRA"

# Normalization dataset
# =================================================================================================
NORM_DATASET_PATH  = "HuggingFaceH4/ultrachat_200k"
NORM_DATASET_SPLIT = "train_sft"

###################################################################################################
# CONFIGS
###################################################################################################

model_config = ModelConfig(
    torch_dtype               = "bfloat16",
    load_in_8bit              = False,
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_use_double_quant = True,
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

quantization_config = get_quantization_config(model_config)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH, 
    num_labels = 1,
    quantization_config = quantization_config,
    device_map = "auto",
    torch_dtype = model_config.torch_dtype
)

# Sync padding tokens
# =================================================================================================

model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model.config.pad_token_id = tokenizer.pad_token_id


###################################################################################################
# Normalization
###################################################################################################
