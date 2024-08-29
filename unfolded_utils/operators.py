import torch
import numpy as np

def inpainting_matrix(shape, prop, seed = 2349231123):
    """
        Return an inpainting matrix with given shape and prop observed pixels.
        Keep seed fixed to ensure consistency of generated matrix.
    """
    len = int(np.prod(shape))
    rng = np.random.default_rng(seed)
    num_obs = int(len*prop)
    obs = rng.choice(len, replace = False, size = num_obs).astype(int)
    A = np.zeros(len)
    A[obs] = 1
    return torch.tensor(A, dtype = torch.float32).reshape(shape)

def build_gaussian_filter(width, sd): # Build gaussian filter with specified width and sd
    # Construct gaussian blurring kernel
    midpt = np.ones(2)*width//2
    filter = np.zeros((width, width), dtype = np.float32)
    for i in range(width):
        for j in range(width):
            filter[i,j] = np.float32(np.exp(-np.sum((np.array([i,j]) - midpt)**2)/2/sd**2)/2/np.pi/sd**2)
    return torch.tensor(filter, dtype = torch.float32).reshape((1,1,width,width))
