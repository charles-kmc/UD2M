import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import copy


# The following code is used to parametrise the layer
class LoRaParametrisation(nn.Module):
    def __init__(self, in_channel, out_channel, rank = 2, alpha = 1, device = "cpu"):
        super(LoRaParametrisation, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.rank = rank
        self.device = device
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.zeros(self.in_channel, self.rank, device = self.device))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_channel, device = self.device))
        nn.init.normal_(self.lora_A, mean = 0, std = 1)
        nn.init.zeros_(self.lora_B)
        self.scale = self.alpha / self.rank
        self.enabled = True
        
    def forward(self, original_weights):
        if self.enabled:
            return original_weights + torch.matmul(self.lora_A, self.lora_B).view(original_weights.shape) * self.scale
        return original_weights

def layer_parametrisation(layer, rank = 2, device = "cpu"):
    in_channel, out_channel = layer.weight.squeeze().shape
    return LoRaParametrisation(in_channel, out_channel, rank = rank, device = device)

def apply_lora_parametrization(model, rank, device):
    #   loop through all the layers of the model
    for name, layer in model.named_modules():
    #    Check if the layer name matches "to_k", "to_q", or "to_v"
        if name.endswith("qkv") or name.endswith("proj_out"):
            P.register_parametrization(layer, "weight", layer_parametrisation(layer, rank=rank,  device=device))           

def LoRa_model(model, rank = 2, device = "cpu"):
    # Apply LoRA parametrization to input blocks
    apply_lora_parametrization(model, rank, device)
    print(f'Number of Layers frozen: {num_frozen_layers(model)}')
    total_parameters_non_lora, total_parameters_lora = total_parameters(model)
    print(f"Total trainable parameters of the model: {total_parameters_non_lora} (non-LoRa) vs {total_parameters_lora} (LoRa) Ratio: {(total_parameters_lora/total_parameters_non_lora)*100:.2f}% of the original model")
    return model

def total_parameters(model):
    n_frozen = 0
    n_lora = 0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            n_frozen += param.numel()
        else:
            n_lora += param.numel()
    return n_frozen, n_lora

# get all layers of the model
def enable_disable_lora(model, enabled = True):
    for _, layer in model.named_children():
        if len(list(layer.children())) > 0:
            enable_disable_lora(layer)
        else:
            for name, _ in layer.named_parameters():
                if "lora_" in name:
                    layer.parametrizations["weight"][0].enabled = enabled  
                    
# number of frozen layers
def num_frozen_layers(model):
    num = 0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
            num += 1
    return num

def make_copy_model(model):
    # Make a deep copy of the model
    return copy.deepcopy(model)