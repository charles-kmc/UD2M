import torch
import kornia
import math
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf as autocorr
from utils.utils_sisr import p2o

# --- Convolution
def actualconvolve(input, kernel, transpose = False):
    if len(kernel.shape) == 2:
        kernel = kernel.unsqueeze(0)
    else:
        pass
    print(kernel.shape)
    if transpose:
        return(kornia.filters.filter2d(input,kernel.rot90(2,[2,1]), border_type='circular'))
    else:
        return(kornia.filters.filter2d(input,kernel, border_type='circular'))
    
# --- kernel to image size
def kernel2Image(kernel, im_shape):
    kernel = kernel.squeeze()
    if len(kernel.shape) > 2:
        raise ValueError("The kernel dimension is not correct")
    device = kernel.device
    N, M = im_shape[-2:]
    n, m = kernel.squeeze().shape
    H = torch.zeros(N, M, dtype=torch.float).to(device)
    H[:n,:m] = kernel
    
    for axis, axis_size in enumerate(kernel.shape):
        H = torch.roll(H, -int((axis_size) / 2), dims=axis)

    FFT_H = torch.fft.fft2(H)
 
    return FFT_H


def ker2img(kernel, im_shape):
    otf = torch.zeros(kernel.shape[:-2] + im_shape)
    otf[...,:kernel.shape[2],:kernel.shape[3]] = kernel
    for axis, axis_size in enumerate(kernel.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    FFT_H = torch.fft.fft2(otf)
    return FFT_H

# --- Forwad operator and its transpose: Hx tHx
def forwad_operator_xh(x, h, transpose = False):
    im_shape = x.shape

    FFT_H = kernel2Image(h, im_shape)
    if len(FFT_H.shape) == 2:
        FFT_H = FFT_H.unsqueeze(0).unsqueeze(0)
        FFT_H = FFT_H.repeat(1,x.shape[1],1,1)
    elif len(FFT_H.shape) == 3:
        FFT_H = FFT_H.unsqueeze(0)
        FFT_H = FFT_H.repeat(1,x.shape[1],1,1)
    elif len(FFT_H.shape) == 4 and FFT_H.shape[1] == 1:
        FFT_H = FFT_H.repeat(1,x.shape[1],1,1)
        pass
    else:
        #raise ValueError("The kernel dimension is not correct")
        pass
    
    FFT_x = torch.fft.fft2(x)

    if transpose:
        FFT_HC = torch.conj(FFT_H)
        return torch.real(torch.fft.ifft2(FFT_HC * FFT_x))
    else:
        return torch.real(torch.fft.ifft2(FFT_H * FFT_x))
    
# --- Forwad operator and its transpose: Hx tHx
def forwad(x, h, transpose = False):
    im_shape = x.shape
    
    FFT_H = p2o(h[None,None,...], im_shape[-2:])
    
    if len(FFT_H.shape) == 2:
        FFT_H = FFT_H.unsqueeze(0).unsqueeze(0)
        FFT_H = FFT_H.repeat(1,x.shape[1],1,1)
    elif len(FFT_H.shape) == 3:
        FFT_H = FFT_H.unsqueeze(0)
        FFT_H = FFT_H.repeat(1,x.shape[1],1,1)
    elif len(FFT_H.shape) == 4 and FFT_H.shape[1] == 1:
        FFT_H = FFT_H.repeat(1,x.shape[1],1,1)
        pass
    else:
        #raise ValueError("The kernel dimension is not correct")
        pass
    
    FFT_x = torch.fft.fft2(x, dim = (-2,-1))

    if transpose:
        FFT_HC = torch.conj(FFT_H)
        return torch.real(torch.fft.ifft2(FFT_HC * FFT_x, dim = (-2,-1)))
    else:
        return torch.real(torch.fft.ifft2(FFT_H * FFT_x, dim = (-2,-1)))

# ------------------------------------------------------------   
#  --- PSF to image size
# ------------------------------------------------------------
def psf_to_image(psf, shape):

    device = psf.device
    N, M = shape[-2:]
    n, m = psf.squeeze().shape

    H = torch.zeros(N, M).to(device)

    H[:n,:m] = psf
    
    for axis, axis_size in enumerate(psf.shape):
        H = torch.roll(H, -int((axis_size) / 2), dims=axis)

    FFT_H = torch.fft.fft2(H)
    return FFT_H

# ------------------------------------------------------------
# --- Convolution operator
# ------------------------------------------------------------
def conv_operator(x, H_FFT, transpose = False):
    if len(H_FFT.shape) == 2:
        H_FFT = H_FFT.unsqueeze(0).unsqueeze(0)
    elif len(H_FFT.shape) == 3:
        H_FFT = H_FFT.unsqueeze(0)
    elif len(H_FFT.shape) == 4:
        pass
    else:
        raise ValueError("The kernel dimension is not correct")
    
    H_FFT = H_FFT.repeat(1,x.shape[1],1,1)
    FFT_x = torch.fft.fft2(x, dim = (-2,-1))

    if transpose:
        FFT_HC = torch.conj(H_FFT)
        return torch.real(torch.fft.ifft2(FFT_HC * FFT_x, dim = (-2,-1)))
    else:
        return torch.real(torch.fft.ifft2(H_FFT * FFT_x, dim = (-2,-1)))

# ------------------------------------------------------------
# --- noise level evaluation function
# ------------------------------------------------------------
def sigma_eval(x, A, snr):
    dimX = torch.numel(x)
    sigma_val = torch.linalg.matrix_norm(A(x)-torch.mean(A(x)), ord='fro')/math.sqrt(dimX*10**(snr/10))
    return sigma_val

# ------------------------------------------------------------
# --- projection oerator
# ------------------------------------------------------------
def projbox(x, min_x, max_x):
    return torch.clamp(x, min_x, max_x)

# ------------------------------------------------------------
# --- welford class
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# --- prox operator of the log-Barrier function
# ------------------------------------------------------------
def prox_LBF(h,mu):
    temp = (h + torch.sqrt(h * h + 4 * mu) ) / 2
    return temp / torch.sum(temp)

# ------------------------------------------------------------
# --- L1 norm
# ------------------------------------------------------------
def l1_norm(h):
    val = h.reshape(-1,1).abs().sum()  
    return val