import wandb

import torch

from transformers import TrainerCallback

# Log metric between initial current and initial weight for given layers
# -------------------------------------------------------------------------------------------------

class WeightChangeCallback(TrainerCallback):
    def __init__(self, layers_to_track=None):
        self.initial_weights = {}
        self.layers_to_track = layers_to_track or []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        for name, param in model.named_parameters():
            if any(layer in name for layer in self.layers_to_track):
                self.initial_weights[name] = param.clone().detach().cpu()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        weight_changes = {}
        for name, param in model.named_parameters():
            if any(layer in name for layer in self.layers_to_track) and param.requires_grad:
                change = torch.norm(param - self.initial_weights[name].to(param.device))
                weight_changes[f"weight_change/{name}"] = change.item()

        wandb.log(weight_changes, step=state.global_step)
