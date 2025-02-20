from __future__ import annotations

import typing as tp
from enum import Enum

import copy

import wandb

from dataclasses import dataclass

import torch
from torch import nn

from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from trl.data_utils import is_conversational


# =================================================================================================
# Helper Functions
# =================================================================================================

# Copy and freeze model
# =================================================================================================

def frozen_copy(model: nn.Module) -> nn.Module:
    """
    Get a frozen copy of a model.
    """
    frozen = copy.deepcopy(model)

    for param in frozen.parameters():
        param.requires_grad = False

    return frozen


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


# Prepare dataset
# =================================================================================================

class DatasetFormat(str, Enum):
    PLAIN = "plain",
    CONVERSATIONAL = "conversational"


# Apply chat template
# -------------------------------------------------------------------------------------------------

def apply_chat_template(
    element,
    tokenizer: PreTrainedTokenizer,
    columns_to_apply_to: list[str],
    dataset_format: DatasetFormat,
    add_generation_prompt: bool = False,
    new_columns: list[str] | None = None,
    system_prompt: str | None = None
):
    if new_columns is None:
        new_columns = columns_to_apply_to
    
    for column_name, new_column_name in zip(columns_to_apply_to, new_columns):
        if dataset_format is DatasetFormat.CONVERSATIONAL:
            prompt = element[column_name]
        elif dataset_format is DatasetFormat.PLAIN:
            if system_prompt is not None:
                prompt = [
                    {'role': "system", 'content': system_prompt},
                    {"role": "user", "content": element[column_name]}
                ]
            else:
                prompt = [
                    {'role': "user", "content": element[column_name]}
                ]
            
        element[new_column_name] = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )

    return element

# Tokenize
# -------------------------------------------------------------------------------------------------

def tokenize(
    element, 
    tokenizer: PreTrainedTokenizer,
    columns_to_apply_to: list[str],
    columns_for_ids: list[str],
    columns_for_attn: list[str],
    add_special_tokens = True,
):
    for column_name, ids_column_name, attn_column_name in zip(
        columns_to_apply_to, columns_for_ids, columns_for_attn
    ):
        tokenized = tokenizer(
            element[column_name], add_special_tokens=add_special_tokens
        )
        element[ids_column_name] = tokenized["input_ids"]
        element[attn_column_name] = tokenized["attention_mask"]

    return element

# Cat column contents
# -------------------------------------------------------------------------------------------------

def cat_columns_contents(
    element,
    lhs_column_names: list[str],
    rhs_column_names: list[str],
    cat_column_names: list[str]
):
    for lhs_column_name, rhs_column_name, cat_column_name in zip(
        lhs_column_names, rhs_column_names, cat_column_names
    ):
        element[cat_column_name] = element[lhs_column_name] + element[rhs_column_name]
        
    return element


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


# =================================================================================================
# Policies mixture class
# =================================================================================================

class PolicyMixture(nn.Module):
    """
    Weighted normalized sum of policies. Policies contribute proportionally to
    coefficients given.
    """

    def __init__(
        self,
        policies: tp.Sequence[nn.Module],
        coefs: tp.Sequence[float]
    ) -> None:

        super().__init__()

        self._policies = nn.ModuleList(policies)
        self._coefs = coefs

    def forward(self, *args, **kwargs):
        logits = []
        for policy, coef in zip(self._policies, self._coefs):
            logits.append(policy(*args, **kwargs).logits * coef)
        res_logits = sum(logits) / sum(self._coefs)

        return CausalLMOutputWithPast(logits=res_logits)


# =================================================================================================
# Policy Commutator
# =================================================================================================

Matrix = tp.Sequence[tp.Sequence[float]]


class PolicyCommutator:
    """
    Policy mixture dispatcher for a set of policies according to commutation
    matrix.
    """

    def __init__(
        self,
        policies: tp.Sequence[nn.Module],
        commutant: Matrix,
    ) -> None:

        assert len(commutant) == len(policies)
        # warning: 2nd commutant dimension validity not ensured

        self._policies = policies
        self._commutant = commutant

    def __getitem__(self, idx: int) -> PolicyMixture:
        return self.get_reference_policy(idx)

    def get_reference_policy(self, idx: int) -> PolicyMixture:
        return PolicyMixture(self._policies, self._commutant[idx])


# =================================================================================================
# Wandb Session Context Manger
# =================================================================================================

class WandbSessionManager:
    def __init__(self, num_sessions: int) -> None:
        self._run_id_map: list[tp.Optional[int]] = num_sessions * [None]

    def __getitem__(self, idx: int) -> WandbSessionBuilder:
        return WandbSessionBuilder(
            manager=self,
            idx=idx,
            run_id=self._run_id_map[idx],
        )

    def __setitem__(self, idx: int, run_id: int) -> None:
        self._run_id_map[idx] = run_id


class WandbSessionBuilder:
    def __init__(
        self,
        manager: WandbSessionManager,
        idx: int,
        run_id: tp.Optional[None]
    ) -> None:
        self._manager = manager
        self._idx = idx
        self._run_id = run_id

    def __call__(self, name: tp.Optional[str] = None) -> WandbSession:
        return WandbSession(
            manager=self._manager,
            idx=self._idx,
            run_id=self._run_id,
            name=name
        )


class WandbSession:
    """
    Wandb Preemptible Session Context Manager.
    """

    def __init__(
        self,
        manager: WandbSessionManager,
        idx: int,
        run_id: tp.Optional[int] = None,
        name: tp.Optional[str] = None,
    ) -> None:
        self._manager = manager
        self._idx = idx
        self._run_id = run_id
        self._name = name

    def __enter__(self) -> tp.Self:
        run = wandb.init(
            id=self._run_id,
            name=self._name,
            resume="allow"
        )
        self._manager[self._idx] = run.id
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        wandb.finish()
