from __future__ import annotations

import typing as tp

import time
from tqdm import tqdm

from dataclasses import dataclass

import torch
from torch import nn

from transformers import (
    Trainer, AutoModelForCausalLM, AutoTokenizer, pipeline
)


# =================================================================================================
# Helper Functions
# =================================================================================================

# Create optimizer with different lerning rates for different parameters (e.g. LoRA and head)
# =================================================================================================

@dataclass
class OptimizerConfig:
    optimizer_type: type
    layer_lr: dict[str, float]
    scheduler: type

def custom_optimizer(model: nn.Module, config: OptimizerConfig):
    layer_params = {layer_name: [] for layer_name in config.layer_lr}
    for params_name, params in model.named_parameters():
        for layer_name in layer_params:
            if layer_name in params_name and params.requires_grad:
                layer_params[layer_name].append(params)

    optimizer_grouped_parameters = [
        {"params": layer_params[layer_name], "lr": config.layer_lr[layer_name]}
        for layer_name in layer_params
    ]

    return config.optimizer_type(optimizer_grouped_parameters)


# Add bias to a head
# =================================================================================================

def set_bias(
    model: nn.Module, 
    layer_path: str, 
    bias: float, 
    dtype: tp.Optional[torch.dtype] = None
)-> None:

    with torch.no_grad():
        parts = layer_path.split(".")
        layer_name = parts[-1]
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Access the target layer
        layer = getattr(parent, layer_name)
        if layer.bias is not None:
            layer.bias.data.fill_(bias)
        else:
            new_layer = nn.Linear(
                layer.in_features, layer.out_features, bias=True, dtype=dtype
            )
            new_layer.weight.data = layer.weight.data.clone()
            new_layer.bias.data.fill_(bias)
            setattr(parent, layer_name, new_layer)


# Push to hub with retries
# =================================================================================================

def push_to_hub_with_retries(
    trainer: Trainer,
    dataset_name: str,
    max_retries: int = 10,
    delay: int = 5
):
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to push...")
            trainer.push_to_hub(dataset_name=dataset_name)
            print("✅ Push successful!")
            return
        except Exception as e:
            print(f"❌ Push failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
            else:
                print("❌ Maximum retries reached. Push failed.")


# Infer model
# =================================================================================================

def get_responses(
    prompts: list[str],
    model_path: str,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    is_chat_model: bool = False,
) -> list[str]:

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    text_generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map='auto',
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    if is_chat_model:
        inputs = [[{'role': "user", 'content': prompt}] for prompt in prompts]
    else:
        inputs = prompts

    responses = []
    for idx in tqdm(
        range(0, len(inputs), batch_size), desc=f'{model_path} inference'
    ):
        batch = inputs[idx:idx+batch_size]
        responses.extend(text_generator(batch))

    if is_chat_model:
        text_responses = [
            response[0]['generated_text'][-1]['content'] for response in responses
        ]
    else:
        text_responses = [
            response[0]['generated_text'] for response in responses
        ]

    return text_responses
