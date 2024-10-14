from torchmetrics.image import StructuralSimilarityIndexMeasure 
import torch 
import math
import os 
import logging
from torchviz import make_dot 
import lpips



class Metrics():
    def __init__(self, device):
        self.device = device
        self.mse = torch.nn.MSELoss()
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.l1_loss = torch.nn.L1Loss().to(device)
    # --- mse
    def mse_function(self,x,y):
        mse_val = self.mse(x,y)
        return mse_val.cpu().detach().squeeze().numpy()

    # --- psnr
    def psnr_function(self,x,y):
        psnr_val =-10*torch.log10(self.mse(x,y))
        return psnr_val.cpu().detach().squeeze().numpy()
    
    # --- ssim
    def ssim_function(self,x,y):
        ssim_val  = self.ssim(x, y)
        return ssim_val.cpu().detach().squeeze().numpy()
    
    # --- L1 loss
    def L1loss(self, x, y):
        l1_val = self.l1_loss(x,y)
        return l1_val.cpu().detach().squeeze().numpy()
    
    def lpips_function(self, x, y):
        lpips_val = self.loss_fn_vgg(x,y)
        return lpips_val.cpu().detach().squeeze().numpy()
