import os.path
import numpy as np
import torch

from DPIR.utils import utils_model
from DPIR.utils import utils_pnp as pnp
from DPIR.utils import utils_sisr as sr
from DPIR.utils import utils_image as util
from DPIR.metrics import *

class DPIR_deb:
    """_summary_
    """
    def __init__(self, model_name = "drunet_color", device = "cpu", x8 = True, modelSigma1 = 49, noise_val = 0.05, pretrained_pth = None):
        self.model_name = model_name
        self.x8 = x8
        self.modelSigma1 = modelSigma1
        self.noise_val = noise_val
        self.modelSigma2 = self.noise_val * 255
        self.sf = 1
        self.device = device
        n_channels = 3 if "color" in self.model_name else 1

        if pretrained_pth is None:
            raise ValueError("Please provide the checkpoint path !!")
        else:
            model_path = os.path.join(pretrained_pth, self.model_name+'.pth')
        
        # load model
        if 'drunet' in self.model_name:
            from DPIR.models.network_unet import UNetRes as net
            self.model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
            self.model.load_state_dict(torch.load(model_path), strict=True)
            self.model.eval()
            self._freeze_model()
            self.model = self.model.to(self.device)
        elif 'ircnn' in model_name:
            from DPIR.models.network_dncnn import IRCNN as net
            self.model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
            self.model25 = torch.load(model_path)
            self.former_idx = 0
        else:
            raise ValueError(" Provide a correct model name !!")        
          
    # run dpir 
    def run(self, y:torch.tensor, k:torch.tensor, iter_num:int = 8):
        """Running DPIR slover for deblurring problems

        Args:
            y (torch.tensor): observation
            k (torch.tensor): blur kernel
            iter_num (int, optional): number of iteration. Defaults to 8.

        Returns:
            x (torch.tensor): restored image
        """
        self.iter_num = iter_num
        
        # set sigmas and rhos      
        self._get_sigmas_rhos()
        
        with torch.no_grad():
            self.iter_num = iter_num
            # initialize x, and pre-calculation
            k = k.unsqueeze(0).unsqueeze(0) if len(k.shape) == 2 else k
            y = y.unsqueeze(0) if len(y.shape) == 3 else y
            [k, y] = util.todevice([k, y], self.device)
            FB, FBC, F2B, FBFy = sr.pre_calculate(y, k, self.sf)
            x = y.clone()
            taus = [self.rhos[i].float().repeat(1, 1, 1, 1) for i in range(iter_num)]
            
            
            # main iterations
            for i in range(iter_num):
                # step 1, FFT
                tau = taus[i]
                x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, self.sf)

                if 'ircnn' in self.model_name:
                    current_idx = int(np.ceil(self.sigmas[i].cpu().numpy() * 255. / 2.) - 1)
                    if current_idx != self.former_idx:
                        self.model.load_state_dict(self.model25[str(current_idx)], strict=True)
                        self.model.eval()
                        self._freeze_model()
                        self.model = self.model.to(self.device)
                    self.former_idx = current_idx

                # step 2, denoiser
                if 'ircnn' in self.model_name:
                    current_idx = int(np.ceil(self.sigmas[i].cpu().numpy()*255./2.)-1)

                    if current_idx != self.former_idx:
                        self.model.load_state_dict(self.model25[str(current_idx)], strict=True)
                        self.model.eval()
                        self._freeze_model()
                        self.model = self.model.to(self.device)
                    self.former_idx = current_idx

                if self.x8:
                    aug_mode = i % 8
                    x = util.augment_img_tensor4(x, aug_mode)

                if 'drunet' in self.model_name:
                    x = torch.cat((x, self.sigmas[i].float().repeat(x.shape[0], 1, x.shape[2], x.shape[3])), dim=1)
                    x = utils_model.test_mode(self.model, x, mode=2, refield=32, min_size=256, modulo=16)
                elif 'ircnn' in self.model_name:
                    x = self.model(x)

                if self.x8:
                    if aug_mode in [3, 5]:
                        x = util.augment_img_tensor4(x, 8 - aug_mode)
                    else:
                        x = util.augment_img_tensor4(x, aug_mode)                         
        return x

    # get hyper parameters  
    def _get_sigmas_rhos(self):
        """Get rhos and sigmas used in DPIR
        """
        rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255/255., self.noise_val), iter_num=self.iter_num, modelSigma1=self.modelSigma1, modelSigma2=self.modelSigma2, w=1.0)
        self.rhos, self.sigmas = torch.tensor(rhos).to(self.device), torch.tensor(sigmas).to(self.device)
    
    # frozen parameters
    def _freeze_model(self):
        for _, v in self.model.named_parameters():
            v.requires_grad = False
        
# dpir solver
def dpir_solver(y:torch.tensor, k:torch.tensor, DPIR:DPIR_deb, iter_num:int = 8):
    """_summary_

    Args:
        y (torch.tensor): observation
        k (torch.tensor): blur kernel
        DPIR (DPIR_deb): DPIR module
        iter_num (int, optional): number of iteration. Defaults to 8.

    Returns:
        sol: output of the solver
    """
    return (DPIR.run(y, k, iter_num = iter_num))

# module for batch
def dpir_module(batch:torch.tensor, k_op:dict, DPIR:DPIR_deb, device:torch.device, iter_num:int = 2):
    ks = k_op["blur_value"]
    noise_vals = k_op["sigma"]
    sol = []
    with torch.no_grad():
        for ii, y in enumerate(batch):
            k = ks[ii]
            x = dpir_solver(y, k, DPIR, iter_num = iter_num)
            sol.append(torch.clamp(x, 0.0, 1.0).to(device))
    del ks
    del noise_vals
    return torch.cat(sol)
        
