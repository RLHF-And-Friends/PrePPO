import typing as tp

from torch import nn

import copy
import wandb

from transformers.modeling_outputs import CausalLMOutputWithPast


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


