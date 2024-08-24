import torch 
import torch.nn as nn 
from degradation_model.utils_deblurs import blur2operator
from utils.utils import image_transform, inverse_image_transform


class Pipeline(nn.Module):
    def __init__(self, lora_model, device = "cpu", dtype = torch.float):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.weight_para = nn.Parameter(torch.ones((3, 256, 256), dtype = self.dtype, device = self.device, requires_grad=True))
        self.l_model = lora_model.to(device)
    def forward(self, x, t, y):
        assert x.shape == y.shape, "x and y must have same shape!!"
        x = self.weight_para * x + (1 - self.weight_para) * y
        out = self.l_model(x, t)
        return out
    
    # -- Consistency loss
    def consistency_loss(self, xpred, y, list_op_blur):
        
        mse_function = nn.MSELoss()
        device = xpred.device
        
        # Create a list of kernels and images
        kernels = list_op_blur["blur_value"] 
        sigmas = list_op_blur["sigma"].to(device)
        mse_per_sample = []
        
        # Loop over kernels and xpreds in parallel
        for i in range(kernels.shape[0]):
            kernel = kernels[i,...]
            xpred_i = inverse_image_transform(xpred[i,...])
            xpred_fft = torch.fft.fft2(xpred_i, dim=(-2, -1))
            FFT_H = blur2operator(kernel, xpred_i.shape[-1]).to(device)
            blurred_fft = FFT_H * xpred_fft
            ax_pred = torch.real(torch.fft.ifft2(blurred_fft, dim=(-2, -1)))
            mse_per_sample.append(mse_function(ax_pred, y[i]).item())
        
        mse_val = torch.tensor(mse_per_sample).to(device)
        loss =  mse_val / (2*sigmas**2)
        
        return loss.mean()