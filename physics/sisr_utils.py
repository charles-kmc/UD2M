import torch.fft
import torch
import kornia

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp2d
import torch.nn.functional as F


def actualconvolve(input, kernel):
    while len(kernel.shape)<3:
        kernel = kernel.unsqueeze(0)
    return(kornia.filters.filter2d(input, kernel, border_type='circular'))

def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b

def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH, image or kernel
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)
    return x

def shift_pixel_batch(x, sf, upper_left=True):
    """
    Shift pixel for super-resolution with different scale factors in a batched tensor.

    Args:
        x: Tensor of shape B*C*W*H
        sf: Scale factor (float)
        upper_left: Shift direction (bool). If True, shift towards upper-left; otherwise, lower-right.

    Returns:
        Tensor of shape B*C*W*H with shifted pixels.
    """
    while x.dim() < 4:
        x = x.unsqueeze(0)
    B, C, W, H = x.shape
    shift = (sf - 1) * 0.5

    # Generate a grid for sampling
    x_coords = torch.linspace(0, W - 1, W, device=x.device)  # X-coordinates
    y_coords = torch.linspace(0, H - 1, H, device=x.device)  # Y-coordinates

    if upper_left:
        x_coords = x_coords + shift
        y_coords = y_coords + shift
    else:
        x_coords = x_coords - shift
        y_coords = y_coords - shift

    # Clip coordinates to remain within bounds
    x_coords = x_coords.clamp(0, W - 1)
    y_coords = y_coords.clamp(0, H - 1)

    # Create a meshgrid for sampling
    grid_x, grid_y = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid = torch.stack((grid_y, grid_x), dim=-1)  # Shape: (W, H, 2)
    grid = grid.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1)  # Shape: (B, C, W, H, 2)
 
    # Normalize the grid for F.grid_sample
    grid = grid.clone()  # Ensure independent memory
    grid[..., 0] = (grid[..., 0] / (H - 1)) * 2 - 1  # Normalize y to [-1, 1]
    grid[..., 1] = (grid[..., 1] / (W - 1)) * 2 - 1  # Normalize x to [-1, 1]

    # Perform the grid sampling
    x = x.view(B * C, 1, W, H)  # Flatten batch and channels
    shifted = F.grid_sample(x, grid.view(B * C, W, H, 2), mode='bilinear', align_corners=True)
    shifted = shifted.view(B, C, W, H)  # Reshape back to original batch format

    return shifted