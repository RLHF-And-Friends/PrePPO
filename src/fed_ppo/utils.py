from __future__ import annotations
import typing as tp
import copy
import wandb

from dataclasses import dataclass

from torch import nn

from datasets import Dataset

from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from trl.data_utils import is_conversational


# =================================================================================================
# Helper Functions
# =================================================================================================

# Copy and freeze model
# -------------------------------------------------------------------------------------------------

def frozen_copy(model: nn.Module) -> nn.Module:
    """
    Get a frozen copy of a model.
    """
    frozen = copy.deepcopy(model)

    for param in frozen.parameters():
        param.requires_grad = False

    return frozen


# Create optimizer with different lerning rates for different parameters (e.g. LoRA and head)
# -------------------------------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    optimizer_type: type
    layer_lr: dict[str, float]


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
# -------------------------------------------------------------------------------------------------

def apply_chat_template(element, tokenizer: PreTrainedTokenizer):
    # PPO dataset
    # ---------------------------------------------------------------------------------------------
    if "prompt" in element.keys():
        if is_conversational(element):
            prompt = element["prompt"]
        else:
            prompt = [{"role": "user", "content": element["prompt"]}]

        element["prompt"] = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt = True, # gen prompt is needed
            tokenize = False,
        )

    # Preference dataset
    # ---------------------------------------------------------------------------------------------
    if "chosen" in element.keys() and "rejected" in element.keys():
        # Apply chat template only if preference dataset is conversational
        if is_conversational(element):
            element["chosen"] = tokenizer.apply_chat_template(
                element["chosen"],
                add_generation_prompt = False, # do not need gen prompt here
                tokenize = False,
            )
            element["rejected"] = tokenizer.apply_chat_template(
                element["rejected"],
                add_generation_prompt = False, # do not need gen prompt here
                tokenize = False,
            )

    return element


def tokenize(element, tokenizer: PreTrainedTokenizer):
    # PPO dataset
    # ---------------------------------------------------------------------------------------------
    if "prompt" in element.keys():
        tokenized = tokenizer(element["prompt"], add_special_tokens=False)
        
        element["input_ids"] = tokenized["input_ids"]

    # Preference dataset
    # ---------------------------------------------------------------------------------------------
    if "chosen" in element.keys() and "rejected" in element.keys():
        chosen_tokenized = tokenizer(element["chosen"], add_special_tokens=False)
        rejected_tokenized = tokenizer(element["rejected"], add_special_tokens=False)
        
        element["input_ids_chosen"] = chosen_tokenized["input_ids"]
        element["attention_mask_chosen"] = chosen_tokenized["attention_mask"]
        element["input_ids_rejected"] = rejected_tokenized["input_ids"]
        element["attention_mask_rejected"] = rejected_tokenized["attention_mask"]
        
    
    return element


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
