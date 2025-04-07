from inference.DiffPIR_original.DiffPIR_models import DiffPIR
from inference.DiffPIR_original.args_diffpir import args_dict
import datetime

import torch

def inference_deb(noise_val, blur_mode):
    device = torch.device("cuda" if torch.is_available() else "cpu")
    args = args_dict()
    frozen_model_ckpt = "diffusion_ffhq_10m.pt"
    epoch = 5
    model_type = None
    cond_used = True
    
    # Solver for deblurring
    diffpir_deb_module = DiffPIR(frozen_model_ckpt, device, args, cond_used = cond_used, epoch= epoch, model_type = model_type)
    
    # run the deblurring solver
    diffpir_deb_module.run(noise_val, blur_mode)
    
    
