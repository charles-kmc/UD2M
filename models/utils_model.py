import torch

import blobfile as bf 
import yaml

def extract_tensor(
        arr:torch.Tensor, 
        timesteps:torch.Tensor, 
        broadcast_shape:torch.Size
    ) -> torch.Tensor:
    out = arr.to(device=arr.device)[timesteps].float()
    while len(out.shape) < len(broadcast_shape):
        out = out[..., None]
    return out.expand(broadcast_shape)

# load yaml file 
def load_yaml(save_checkpoint_dir, epoch, date, task):
    dir_ = bf.join(save_checkpoint_dir, date)
    filename = f"LoRA_model_{task}_{(epoch):03d}.yaml"
    filepath = bf.join(dir_, filename)
    with open(filepath, 'r') as f:
        args = yaml.safe_load(f)
    
    return args
