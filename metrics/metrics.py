from torchmetrics.image import StructuralSimilarityIndexMeasure 
import torch 
import math
import os 
import logging
from torchviz import make_dot 
import lpips
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 

class Metrics():
    def __init__(self, device):
        self.device = device
        self.mse = torch.nn.MSELoss()
        self.loss_fn_vgg = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
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
        x = x.unsqueeze(0) if x.dim()==3 else x
        y = y.unsqueeze(0) if y.dim()==3 else y
        ssim_val  = self.ssim(x, y)
        return ssim_val.cpu().detach().squeeze().numpy()
    
    # --- L1 loss
    def L1loss(self, x, y):
        l1_val = self.l1_loss(x,y)
        return l1_val.cpu().detach().squeeze().numpy()
    
    def lpips_function(self, x, y):
        x = 2*x - 1
        y = 2*y - 1
        x,y = self.get_4d_tensor([x,y])
        lpips_val = self.loss_fn_vgg(x,y)
        return lpips_val.cpu().squeeze().numpy()
    
    def get_4d_tensor(self, L):
        L_tensor = []
        for x in L:
            while x.dim() != 4:
                x = x.unsqueeze(0)
            L_tensor.append(x)
        
        return L_tensor
