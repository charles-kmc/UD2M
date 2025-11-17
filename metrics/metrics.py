from torchmetrics.image import StructuralSimilarityIndexMeasure 
import torch 
from torchviz import make_dot 
import lpips
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 
import numpy as np
import os
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class Metrics():
    def __init__(self, device):
        self.device = device
        self.mse = torch.nn.MSELoss()
        self.loss_fn_vgg = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', ).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.l1_loss = torch.nn.L1Loss().to(device)
    # --- mse
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
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
            y = y.repeat(1,3,1,1)
        lpips_val = self.loss_fn_vgg(x,y)
        return lpips_val.cpu().squeeze().numpy()
    
    def get_4d_tensor(self, L):
        L_tensor = []
        for x in L:
            while x.dim() != 4:
                x = x.unsqueeze(0)
            L_tensor.append(x)
        
        return L_tensor
    
    

def psnr(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val


def snr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred) ** 2)
    snr = 10 * np.log10(np.mean(gt ** 2) / noise_mse)

    return snr


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred) ** 2)

    return noise_mse


def ssim(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None, multichannel: bool = False
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    # if not gt.ndim == 3:
    #   raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    if multichannel:
        ssim = structural_similarity(
            gt, pred, data_range=maxval, channel_axis=2
        )
    else:
        ssim = structural_similarity(
            gt, pred, data_range=maxval
        )

    return ssim
