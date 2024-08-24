import numpy as np
import scipy 
import math

import torch

def fspecial_gaussian_tol(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h

def fspecial_gaussian(hsize, sigma):
    center = int((hsize+1)/ 2)
    xker = np.arange(-hsize + center, hsize - center+1).reshape(-1,1)
    yker = xker.T    
    c_exp_term = (xker**2 + yker**2) / sigma**2; const = 1 / (2*math.pi*sigma**2)
    kernel = const * np.exp(-c_exp_term/2)
    kernel_norm = kernel / np.sum(kernel.reshape(-1,1))
    return torch.from_numpy(kernel_norm)