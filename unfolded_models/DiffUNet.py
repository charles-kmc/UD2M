import torch 
import datetime 
import os 
from utils import utils_model
import numpy as np
from models.load_frozen_model import load_frozen_model
from utils_lora.lora_parametrisation import LoRa_model, make_copy_model
import copy 

def load_model(
        device, 
        use_lora = False,
        model_path = './'
    ):
    model_name = 'diffusion_ffhq_10m' #'diffusion_ffhq_10m'  # diffusion_ffhq_10m, 256x256_diffusion_uncond; set diffusino model

    # ----------------------------------------
    # load model
    # ----------------------------------------
    print(model_name)
    model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
        
    args = utils_model.create_argparser(model_config).parse_args([])
    frozen_model, diffusion = load_frozen_model(model_name, model_path, device)

    if use_lora:
        lora_model = copy.deepcopy(frozen_model)
        LoRa_model(lora_model, device)
        for param in lora_model.out.parameters():
            param.requires_grad_(True)
        for name, param in lora_model.named_parameters():
            if "output_blocks.2.0.out_layers.3" in name:
                param.requires_grad_(True)
            if "time_embed" in name:
                param.requires_grad_(True)
        return lora_model
    else: 
        return frozen_model

class DiffUNet(torch.nn.Module):
    def __init__(self, device, use_lora = False, model_path =  './model_zoo/', beta_start = 0.1/1000, beta_end = 20/1000, T=1000):
        super(DiffUNet, self).__init__()
        self.device = device 
        self.explicit_prior = False
        self.T = T
        self.model = load_model(self.device, use_lora=use_lora, model_path=model_path)
        self.betas = np.linspace(beta_start, beta_end, self.T, dtype=np.float32)
        self.betas = torch.from_numpy(self.betas)
        self.alphas                  = 1.0 - self.betas
        self.alphas_cumprod          = np.cumprod(self.alphas.cpu(), axis=0)
        self.sqrt_alphas_cumprod     = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_1m_alphas_cumprod  = torch.sqrt(1. - self.alphas_cumprod).to(device)
        self.sigmas   = torch.div(self.sqrt_1m_alphas_cumprod, self.sqrt_alphas_cumprod)


    def forward(self, x, t, eta = 1.):
        x = 2.0 * x - 1.0  ## Denoiser operates on images in [-1,1]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(list([t]))
        noise_map = self.model(x, t.clone().flatten() * eta)
        noise_map = noise_map[:, :3, ...]   # Take noise map
        x_denoised = x/self.sqrt_alphas_cumprod[t] - self.sigmas[t]*noise_map
        x_denoised = (x_denoised + 1.0)/2.0  # Rescale back to [0,1]
        return x_denoised
    
    def prox(self, x, ts, *args, eta = 1., **kwargs):  # Return prox operator
        return self.forward(x, ts, eta = eta)