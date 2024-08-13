import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import random

# deblurring model
class DeblurringModel(object):
    def __init__(
                self, 
                im_size, 
                blur_sizes = [21], 
                blur_type = ["gaussian", "uniform"], 
                device = "cpu", 
                dtype = torch.float32
            ):
        self.device = device
        self.blur_sizes = blur_sizes
        self.dtype = dtype
        self.im_size = im_size
        self.blur_types = blur_type
        self.noise_std = [0.005, 0.01, 0.05]
        
        self.blur = None
    
    # set seeed  
    def set_seed(self, seed):
        torch.manual_seed(seed)
        
    # get noisy image
    def get_noisy_image(self, x:torch.tensor):
        noise_std = np.random.choice(self.noise_std)
        Ax = self.Ax(x)  
        y = Ax + noise_std * torch.randn_like(Ax)
        op = {
            f"blur_tpye": self.blur_type,
            f"blur_size": self.blur_size,
            f"im_size": self.im_size,
            f"blur_value": self.blur,  
        }
        return y, op
    
    # get blur kernel and operator    
    def _get_kernel(self):
        self.set_seed(random.randint(0,20))
        self.blur_type = np.random.choice(self.blur_types) 
        self.blur_size = np.random.choice(self.blur_sizes)
        if self.blur_type == "gaussian":
            blur = gaussian_kernel(self.blur_size, dtype = self.dtype)
        elif self.blur_type == "uniform":
            blur = uniform_kernel(self.blur_size, dtype = self.dtype)
        elif self.blur_type == "motion":
            blur = motion_kernel(self.blur_size)
            self.blur_size = blur.shape[-1]
        else:
            raise ValueError(f"The blur type: {self.blur_type} is not correct!!")
        
        self.blur = blur
        
        FFT_H = blur2operator(self.blur, self.im_size)
        return blur, FFT_H
    
    # get blurred image
    def Ax(self, x):
        self.blur, self.FFT_H = self._get_kernel()
        self.blur = self.blur.to(self.device)
        self.FFT_H = self.FFT_H.to(self.device)
        
        if len(self.FFT_H.shape) > 2:
            raise ValueError(f"The kernel dimension {self.FFT_H.shape} is not implemented!!")

        assert x.shape[-2:] ==self.FFT_H.shape[-2:], f"The image size {x.shape[-2:]} and the kernel size {self.FFT_H.shape[-2:]} are not the same!!"
        
        return torch.real(
            torch.fft.ifft2(
                self.FFT_H * torch.fft.fft2(x, dim = (-2,-1)), 
                dim = (-2,-1)
                )
            )

# get the operator from the blur kernel
def blur2operator(blur, im_size):
    blur = blur.squeeze()
    if len(blur.shape) > 2:
        raise ValueError("The kernel dimension is not correct")
    device = blur.device
    n, m = blur.squeeze().shape
    H = torch.zeros(im_size,im_size, dtype=blur.dtype).to(device)
    H[:n,:m] = blur
    
    for axis, axis_size in enumerate(blur.shape):
        H = torch.roll(H, -int((axis_size) / 2), dims=axis)
    FFT_H = torch.fft.fft2(H, dim = (-2,-1))
 
    return FFT_H

    # Gaussian function
def gaussian_kernel(hsize:int, dtype = torch.float32, sigma:float = 3.0):
    center = int((hsize+1)/ 2)
    xker = np.arange(-hsize + center, hsize - center+1).reshape(-1,1)
    yker = xker.T    
    c_exp_term = (xker**2 + yker**2) / sigma**2; const = 1 / (2*math.pi*sigma**2)
    kernel = const * np.exp(-c_exp_term/2)
    kernel_norm = kernel / np.sum(kernel.reshape(-1,1))
    return torch.from_numpy(kernel_norm).to(dtype)

# uniform function
def uniform_kernel(hsize, dtype = torch.float32):
    kernel = np.ones((hsize, hsize)) / hsize**2
    return torch.from_numpy(kernel).to(dtype)

# motion function
def motion_kernel(hsize):
    NotImplementedError("The motion kernel is not implemented yet!!")
    
# Tikhonov solution  
def deb_data_solution(y, blur, im_size, device, alpha):
    with torch.no_grad():
        FFT_H = blur2operator(blur, im_size).to(device)
        FFT_HC = torch.conj(FFT_H)
        fft_y = torch.fft.fft2(y, dim = (-2,-1)).to(device)
        FR = FFT_HC * fft_y
        
        div_term = FFT_HC * FFT_H + alpha
    
        sol = torch.real(
            torch.fft.ifft2(
                torch.divide(FR, div_term), 
                dim = (-2,-1)
                )
            )
    return sol.unsqueeze(0)

# batch solution    
def deb_batch_data_solution(batch_y, batch_blur_kernel_op, device, alpha = torch.tensor(0.01)):
    batch_sol = []
    # loop over the batch
    for i in range(len(batch_y)):
        y_i = batch_y[i, :,:,:]
        im_size = batch_blur_kernel_op["im_size"][i]
        blur_i = batch_blur_kernel_op["blur_value"][i]
        # get the Tikhonov solution
        batch_sol.append(deb_data_solution(y_i, blur_i , im_size, device, alpha))
    batch_sol = torch.cat(batch_sol)
    
    return batch_sol