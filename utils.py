import typing as tp

from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

###############################################################################
# Policies mixture class
###############################################################################

class PolicyMixture(nn.Module):
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
    
