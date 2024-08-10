import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from guided_diffusion.unet import *
import copy


# The following code is used to parametrise the layer
class LoRaParametrisation(nn.Module):
    def __init__(
                self, 
                in_channel, 
                out_channel, 
                rank = 2, 
                alpha = 1, 
                device = "cpu"
            ):
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

# The following code is used to parametrise the layer   
def layer_parametrisation(
                        layer, 
                        rank = 2, 
                        device = "cpu"
                    ):
    
    in_channel, out_channel = layer.weight.squeeze().shape
    return LoRaParametrisation(
                        in_channel, 
                        out_channel, 
                        rank = rank, 
                        device = device
                    )

# apply LoRa parametrisation
def apply_lora_parametrization(
                        block, 
                        rank, 
                        device
                    ):
    for block_att in block.children():
        if isinstance(block_att, AttentionBlock):
            # qkv layer followed by lora layer
            conv1d_qkv = block_att.qkv
            P.register_parametrization(
                                conv1d_qkv, 
                                "weight", 
                                layer_parametrisation(
                                    conv1d_qkv, 
                                    rank=rank, 
                                    device=device
                                )
                            )
            # projection layer followed by lora layer
            conv1d_proj_out = block_att.proj_out
            P.register_parametrization(
                                conv1d_proj_out, 
                                "weight", 
                                layer_parametrisation(
                                    conv1d_proj_out, 
                                    rank=rank, 
                                    device=device
                                )
                            )

# The following code is used to apply LoRa to the model
def LoRa_model(
            model, 
            input_block_layer_lora = [10, 14], 
            middle_block_layer_lora = [1], 
            output_block_layer_lora = [3, 6], 
            rank = 2, 
            device = "cpu"
        ):
        
    # Apply LoRA parametrization to input blocks
    for ii, block in enumerate(model.input_blocks):
        if ii in input_block_layer_lora:
            apply_lora_parametrization(block, rank, device)
    
    # Apply LoRA parametrization to the middle block
    assert len(middle_block_layer_lora) == 1, "Middle block has only 1 block with one self attention block."
    if middle_block_layer_lora is not None:
        apply_lora_parametrization(model.middle_block, rank, device)

    # Apply LoRA parametrization to output blocks
    for ii, block in enumerate(model.output_blocks):
        if ii in output_block_layer_lora:
            apply_lora_parametrization(block, rank, device)       
    
    # freeze non-LoRA parameters
    print(f'Number of Layers frozen: {num_frozen_layers(model)}')
    
    total_parameters_non_lora, total_parameters_lora = total_parameters(model)
    print(f"Total trainable parameters of the model: {total_parameters_non_lora} (non-LoRa) vs {total_parameters_lora} (LoRa) Ratio: {(total_parameters_lora/total_parameters_non_lora)*100:.2f}% of the original model")
    return model

# The following code is used to count the number of parameters in the model  
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