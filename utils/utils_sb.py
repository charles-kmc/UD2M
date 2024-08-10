from torchmetrics.image import StructuralSimilarityIndexMeasure # type: ignore
import torch
import math

# --- noise level evaluation function
def sigma_eval(x, A, snr):
    dimX = torch.numel(x)
    sigma_val = torch.linalg.matrix_norm(A(x)-torch.mean(A(x)), ord='fro')/math.sqrt(dimX*10**(snr/10))
    return sigma_val

# --- projection oerator
def projbox(x, min_x, max_x):
    return torch.clamp(x, min_x, max_x)

# --- welford class
class welford:
    def __init__(self, x, startit = 1):
        self.k = startit
        self.M = x.clone().detach()
        self.S = 0

    # -- update new data
    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S += (x - self.M)*(x - Mnext)
        self.M = Mnext

    # -- get the posterior mean
    def get_mean(self):
        return self.M
    
    # -- get the posterior variance
    def get_var(self):
        return self.S/(self.k-1)

# --- Metrics class for evaluation
class Metrics():
    def __init__(self, device):
        self.device = device
        self.mse = torch.nn.MSELoss()
    # --- mse
    def mse_function(self,x,y):
        return self.mse(x,y)

    # --- psnr
    def psnr_function(self,x,y):
        mse_xy =self.mse(x,y)
        return -10*torch.log10(mse_xy)
    
    # --- ssim
    def ssim_function(self,x,y):
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        ss  = ssim(x, y)
        return ss
    
    # --- L1 loss
    def L1loss(self, x, y):
        L1loss = torch.nn.L1Loss()
        return L1loss(x,y)

# --- noise level evaluation function
def sigma_eval(x, A, snr):
    dimX = torch.numel(x)
    sigma_val = (torch.linalg.matrix_norm(
                    A(x)-torch.mean(A(x)), 
                    ord='fro'
                    )/math.sqrt(dimX*10**(snr/10))
                )   
    return sigma_val