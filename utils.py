import typing as tp

import copy

from torch import nn
from trl import PPOTrainer
from transformers.modeling_outputs import CausalLMOutputWithPast


###############################################################################
# Helper Functions
###############################################################################

def freeze(model: nn.Module) -> nn.Module:
    """
    Get a frozen copy of a model.
    """
    frozen = copy.deepcopy(model)

    for param in frozen.parameters():
        param.requires_grad = False

    return frozen


###############################################################################
# Policies mixture class
###############################################################################

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


###############################################################################
# Policy Commutator
###############################################################################

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


###############################################################################
# CustomPPOTrainer
###############################################################################

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

