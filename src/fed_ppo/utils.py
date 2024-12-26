import typing as tp

import copy

from dataclasses import dataclass

from datasets import Dataset

from torch import nn

from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from trl import PPOTrainer


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


# Prepare dataset for PPO trainer
# -------------------------------------------------------------------------------------------------

def apply_chat_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    prompt_field: str = "prompt"
):
    """
    tackle prompt as user chat message
    """
    def foo(element: dict):
        outputs = tokenizer.apply_chat_template([{
            "role": "user",
            "content": element[prompt_field]
        }])
        return {prompt_field: outputs}

    return dataset.map(
        foo,
        batched=True,
        remove_columns=dataset.column_names
    )


def tokenize_prompt(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    prompt_field: str = "prompt"
):
    """
    pre-tokenize the dataset before training; only collate during training
    """

    def tokenize(element):
        outputs = tokenizer(
            element[prompt_field],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )


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
# CustomPPOTrainer
# =================================================================================================

class CustomPPOTrainer(PPOTrainer):
    """
    Slightly modified PPOTrainer.
    """

    def __init__(self, *args, **kwargs):
        """
        Regular init with no cumbersome run renaming.
        """

        super().__init__(*args, **kwargs)

        if hasattr(self.args, "exp_name"):
            self.args.run_name = f"{self.args.exp_name}"
