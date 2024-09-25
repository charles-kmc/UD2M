# from torchmetrics.image import StructuralSimilarityIndexMeasure 
import torch 
import math
import os 
import logging
# from torchviz import make_dot 
import lpips


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
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    # --- mse
    def mse_function(self,x,y):
        return self.mse(x,y)

    # --- psnr
    def psnr_function(self,x,y):
        mse_xy =self.mse(x,y)
        return -10*torch.log10(mse_xy)
    
    # --- ssim
    # def ssim_function(self,x,y):
    #     ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
    #     ss  = ssim(x, y)
    #     return ss
    
    # --- L1 loss
    def L1loss(self, x, y):
        L1loss = torch.nn.L1Loss()
        return L1loss(x,y)
    
    def lpips_function(self, x, y):
        return self.loss_fn_vgg(x,y)

# --- noise level evaluation function
def sigma_eval(x, A, snr):
    dimX = torch.numel(x)
    sigma_val = (torch.linalg.matrix_norm(
                    A(x)-torch.mean(A(x)), 
                    ord='fro'
                    )/math.sqrt(dimX*10**(snr/10))
                )   
    return sigma_val

def image_transform(x):
    return 2*x - 1 

def inverse_image_transform(x):
    return torch.clamp(0.5*x +0.5, 0, 1)

# register Hook for debugging
class RegisterHook:
    def __init__(self):
        pass
    def forward_hook(self, module, input, output):
        print(f"Forward hook: {module.__class__.__name__}")
        print(f"Output: {output.grad_fn}")

    def backward_hook(self, module, grad_input, grad_output):
        print(f"Backward hook: {module.__class__.__name__}")
        print(f"Grad Output: {grad_output}")

    def forward(self, model, object):
    # Register hooks
        for _, module in model.named_modules():
            if isinstance(module, object):
                module.register_forward_hook(self.forward_hook)
                module.register_backward_hook(self.backward_hook)
                
# get the graph and achitecture
# def modelGaph(model, pred, save_path = "."):
#     dot = make_dot(pred, params=dict(model.named_parameters()))

#     # Save the visualized achitecture as a PNG file
#     dot.format = 'png'
#     dot.render('model_architecture')
    
#     # Visualize the computational graph
#     graph = make_dot(pred, params=dict(model.named_parameters()))
#     graph.render("simple_model_graph", format="png")  
#     graph.view()  

# Define the DotDict class
class DotDict(dict):
    """A dictionary that allows dot notation access to keys."""
    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        if item in self:
            del self[item]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
