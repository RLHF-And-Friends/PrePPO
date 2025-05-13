from tqdm import tqdm

from datasets import load_dataset, Dataset

from vllm import LLM, SamplingParams


# Paths and variables
# =================================================================================================

# Initial dataset
# -------------------------------------------------------------------------------------------------
DATASET_PATH = "RLHF-And-Friends/wiki_lingua"
CONFIGURATIONS = ["en", "ru", "de", "fr", "es", "it", "nl"]

TEXT_FIELD = "source"

NUM_SAMPLES = {
    "train": 10000,
    "validation": 1000,
}

# Target dataset
# -------------------------------------------------------------------------------------------------
TARGET_DATASET_PATH = "RLHF-And-Friends/wiki_lingua_paired"

# Generator model
# -------------------------------------------------------------------------------------------------
GENERATOR_PATH = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = (
    "You are a summarization engine. Your only task is to return a short summary of the "
    "input text. Respond with the summary **only**. Do not include any explanations, "
    "prefixes, or extra text."
)

# Make vllm generator
# =================================================================================================

generator = LLM(model=GENERATOR_PATH)

# Make generation parameters
# =================================================================================================

sampling_params_a = SamplingParams(
    temperature=0.2,
    top_p=0.95,
    max_tokens=256,
)

sampling_params_b = SamplingParams(
    temperature=1,
    top_p=0.95,
    max_tokens=256,
)

# Generation
# =================================================================================================

for configuration in tqdm(CONFIGURATIONS, desc="Processing configurations"):

    dataset = load_dataset(DATASET_PATH, configuration)

    for split, split_samples in NUM_SAMPLES.items():
        split_dataset = dataset[split][:split_samples]
        split_texts = split_dataset[TEXT_FIELD]

        split_messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ] for text in split_texts
        ]
        outputs_a = generator.chat(
            split_messages,
            sampling_params=sampling_params_a,
        )
        outputs_b = generator.chat(
            split_messages,
            sampling_params = sampling_params_b,
        )

        generated_summaries_a = [output.outputs[0].text for output in outputs_a]
        generated_summaries_b = [output.outputs[0].text for output in outputs_b]

        new_dataset_split = Dataset.from_dict({
            "text": split_texts,
            "summary_a": generated_summaries_a,
            "summary_b": generated_summaries_b,
        })

        new_dataset_split.push_to_hub(
            repo_id=TARGET_DATASET_PATH,
            config_name=configuration,
            split=split
        )
