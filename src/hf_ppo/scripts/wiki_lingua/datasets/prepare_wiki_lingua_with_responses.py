from vllm import LLM, SamplingParams

from datasets import load_dataset


# Paths and variables
# =================================================================================================

DATASET_PATH = "RLHF-And-Friends/wiki_lingua"
CONFIGURATION = "en"
SPLITS = ["train", "validation", "test"]
TEXT_FIELD = "source"

TARGET_DATASET_PATH = "RLHF-And-Friends/wiki_lingua_preference"

GENERATOR_PATH = "meta-llama/Llama-3.1-8B-Instruct"

# Make vllm generator and judge
# =================================================================================================

generator = LLM(model=GENERATOR_PATH)

# Make generation parameters
# =================================================================================================

sampling_params = SamplingParams(
    temperature=0.9,
    top_p=0.95,
)

# Batched generation
# =================================================================================================

dataset = load_dataset(DATASET_PATH, CONFIGURATION)

for split in SPLITS:
    dataset_split = dataset[split]
    prompts = dataset_split[TEXT_FIELD]

    outputs = generator.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    
    dataset_split["llm_summaries"] = outputs

    dataset_split.push_to_hub(
        repo_id=TARGET_DATASET_PATH,
        config_name=CONFIGURATION,
        split=split
    )
