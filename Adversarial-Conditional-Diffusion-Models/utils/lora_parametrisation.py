import torch#type:ignore
import torch.nn as nn#type:ignore
import torch.nn.utils.parametrize as P #type:ignore
import copy

# The following code is used to parametrise the layer
class LoRaParametrisation(nn.Module):
    def __init__(self, in_channel, out_channel, dtype, rank, alpha = 1.5):
        super(LoRaParametrisation, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dtype = dtype
        self.rank = rank
        self.alpha = alpha
        size_A = (self.in_channel, self.rank)
        size_B = (self.rank, self.out_channel)
        self.lora_A = nn.Parameter(torch.zeros(size=size_A, dtype = self.dtype, requires_grad=True))
        self.lora_B = nn.Parameter(torch.zeros(size=size_B, dtype = self.dtype, requires_grad=True))
        
        # Verify requires_grad
        assert self.lora_A.requires_grad
        assert self.lora_B.requires_grad
        nn.init.xavier_normal_(self.lora_A)
        nn.init.xavier_normal_(self.lora_B)
        # nn.init.normal_(self.lora_A, mean = 0, std = 1)
        # nn.init.zeros_(self.lora_B)
        self.scale = self.alpha / self.rank
        self.enabled = True
        
    def forward(self, original_weights):
        if self.enabled:
            lora_update = torch.matmul(self.lora_A, self.lora_B).view(original_weights.shape) 
            lora_update = lora_update * self.scale
            # print('LW: ', original_weights + lora_update)
            return original_weights + lora_update
        return original_weights

def layer_parametrisation(layer, rank=2):
    in_channel, out_channel = layer.weight.squeeze().shape
    dtype = layer.weight.dtype
    return LoRaParametrisation(
        in_channel, 
        out_channel, 
        dtype, 
        rank = rank
    )

def apply_lora_parametrization(model, target_layer, rank):
    for name, layer in model.named_modules():
    #    Check if the layer name ends with "qkv" or "proj_out"
        if name.endswith(tuple(target_layer)):
            print(f"Applying LoRA to {name}")
            P.register_parametrization(layer, "weight", layer_parametrisation(layer, rank=rank))           

def LoRa_model(model, target_layer, rank = 2):
    # Apply LoRA parametrization to input blocks
    apply_lora_parametrization(model, target_layer, rank)
    print(f'Number of Layers frozen: {num_frozen_layers(model)}')
    total_parameters_non_lora, total_parameters_lora = total_parameters(model)
    print(f"Total trainable parameters of the model: {total_parameters_non_lora} (non-LoRa) vs {total_parameters_lora} (LoRa) Ratio: {(total_parameters_lora/total_parameters_non_lora)*100:.2f}% of the original model")

def total_parameters(model):
    n_frozen = 0
    n_lora = 0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            n_frozen += param.numel()
        else:
            n_lora += param.numel()
    return n_frozen, n_lora

def enable_disable_lora(model, enabled = True):
    for name, _ in model.named_parameters():
        if "lora_" in name:
            layer_name = name.split(".parametrizations")[0]
            submodule = model
            for part in layer_name.split("."):
                submodule = getattr(submodule, part, None)
                if submodule is None:
                    break            
            if submodule and hasattr(submodule, "parametrizations"):
                if "weight" in submodule.parametrizations:
                    parametrization_list = submodule.parametrizations["weight"]
                    for p in parametrization_list:
                        if hasattr(p, "enabled"):
                            p.enabled = enabled
                            print(f"Set enabled to {enabled} for {name}")
                    
                    
                    
# number of frozen layers
def num_frozen_layers(model):
    num = 0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
            num += 1
    return num
