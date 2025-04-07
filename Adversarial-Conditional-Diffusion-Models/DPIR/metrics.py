import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

class Metrics():
    def __init__(self, device):
        self.device = device
    # --- mse
    def mse_function(self,x,y):
        mse = torch.nn.MSELoss()
        return mse(x,y)

    # --- psnr
    def psnr_function(self,x,y):
        mse = torch.nn.MSELoss()
        msexy = mse(x,y)
        return -10*torch.log10(msexy)
    
    def ssim_function(self,x,y):
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        ss  = ssim(x, y)
        return ss
    
    def L1loss(self, x, y):
        L1loss = torch.nn.L1Loss()
        return L1loss(x,y)
    