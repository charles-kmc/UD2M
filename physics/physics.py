import random
import os
from PIL import Image

import torch
import torchvision.transforms as transforms
from .blur_functions import UniformMotionBlurGenerator
from utils.utils import inverse_image_transform

# physic module        
class Deblurring:
    def __init__(
        self, 
        kernel_size:int, 
        blur_name:str, 
        random_blur: bool = False,
        device:str = "cpu",
        sigma_model:float= 0.05, 
        dtype = torch.float32, 
        transform_y:bool = False,
        path_motion_kernel:str = "/users/cmk2000/sharedscratch/Datasets/blur-kernels/kernels",
        mode = "test",
    ) -> None:
        # parameters
        self.transform_y = transform_y
        self.device = device
        self.sigma_model = 2 * sigma_model if self.transform_y else sigma_model
        self.dtype = dtype
        self.kernel_size = kernel_size
        self.blur_name = blur_name
        self.random_blur = random_blur
        self.mode = mode
        self.uniform_kernel_generator = UniformMotionBlurGenerator(kernel_size=(kernel_size,kernel_size))
        
        # motion kernel
        if mode == "train":
            self.path_motion_kernel_train = os.path.join(path_motion_kernel, "training_set")
            self.list_motion_kernel = os.listdir(self.path_motion_kernel_train)
        else:
            self.path_motion_kernel_val = os.path.join(path_motion_kernel, "validation_set")
            self.list_motion_kernel = os.listdir(self.path_motion_kernel_val)
        
        # defined transforms 
        self.transforms_kernel = transforms.Compose([
                transforms.Resize((self.kernel_size, self.kernel_size)),
                transforms.ToTensor()
            ])
        
    def _get_kernel(self):
        if self.blur_name == "gaussian":
            blur_kernel = self._gaussian_kernel(self.kernel_size).to(self.device)
        elif self.blur_name  == "uniform":
            blur_kernel = self._uniform_kernel(self.kernel_size).to(self.device)
        elif self.blur_name  == "motion":
            blur_kernel = self._motion_kernel(self.kernel_size).to(self.device)
        elif self.blur_name == "uniform_motion":
            blur_kernel = self._uniform_motion_kernel().to(self.device)
        else:
            raise ValueError(f"Blur type {self.blur_name } not implemented !!")
        return blur_kernel

    def _uniform_motion_kernel(self):
        return self.uniform_kernel_generator.step(1)["filter"]

    def _motion_kernel(self):
        if self.mode == "train":
            kernel_name = random.choice(self.list_motion_kernel) 
            kernel = Image.open(os.path.join(self.path_motion_kernel_train, kernel_name)) 
        else:
            kernel_name = self.list_motion_kernel[7]  
            kernel = Image.open(os.path.join(self.path_motion_kernel_val, kernel_name))
            
        kernel_ = self.transforms_kernel(kernel)[0,...].squeeze()
        kernel_ /= kernel_.sum()
        return kernel_
    
    def _gaussian_kernel(self, kernel_size, std = 3.0):
        c = int((kernel_size - 1)/2)
        v = torch.arange(-kernel_size + c, kernel_size - c, dtype = self.dtype).reshape(-1,1)
        vT = v.T
        exp = torch.exp(-0.5*(v**2 + vT**2)/std)
        exp /= torch.sum(exp)
        return exp 
        
    def _uniform_kernel(self, kernel_size):
        exp = torch.ones(kernel_size, kernel_size).to(self.dtype)
        exp /= torch.sum(exp)
        return exp
    
    def blur2A(self, im_size):
        if not hasattr(self, "blur"):
            self.blur = self._get_kernel()
        else:
            pass
        
        if len(self.blur.shape) > 2:
            raise ValueError("The kernel dimension is not correct")
        device = self.blur.device
        n, m = self.blur.shape
        H_op = torch.zeros(im_size, im_size, dtype=self.blur.dtype).to(device)
        H_op[:n,:m] = self.blur
        for axis, axis_size in enumerate(self.blur.shape):
            H_op = torch.roll(H_op, -int(axis_size / 2), dims=axis)
        FFT_h = torch.fft.fft2(H_op, dim = (-2,-1))
        return FFT_h[None, None, ...], self.blur

    # forward operator
    def Ax(self, x):
        im_size = x.shape[-1]
        FFT_h, h = self.blur2A(im_size)
        FFT_x = torch.fft.fft2(x, dim = (-2,-1))
        ax = torch.real(
            torch.fft.ifft2(FFT_h * FFT_x, dim = (-2,-1))
        )
        return ax
    
    # forward operator
    def ATx(self, x):
        im_size = x.shape[-1]
        FFT_h, h = self.blur2A(im_size)
        FFT_hc = torch.conj(FFT_h)
        FFT_x = torch.fft.fft2(x, dim = (-2,-1))
        atx = torch.real(
            torch.fft.ifft2(FFT_hc * FFT_x, dim = (-2,-1))
        )
        return atx
    
    # observation y
    def y(self, x:torch.Tensor)->torch.Tensor:
        if not self.transform_y:
            x0 = inverse_image_transform(x)
        else:
            x0 = x
        self.blur = self._get_kernel()
        out = self.Ax(x0) + self.sigma_model * torch.randn_like(x0)
        return out

class Inpainting:
    def __init__(
        self, 
        mask_rate: float,
        im_shape:int,
        mask_type:str = "random",
        device:str = "cpu",
        sigma_model:float= 0.05, 
        pixelwise=True, 
        box_proportion = None, 
        index_ii = None, 
        index_jj = None,
        dtype = torch.float32, 
        transform_y:bool = False,
        mode = "test",
    ) -> None:
        # parameters
        self.im_shape = im_shape
        self.transform_y = transform_y
        self.device = device
        self.sigma_model = 2 * sigma_model if self.transform_y else sigma_model
        self.dtype = dtype
        self.mask_rate = mask_rate
        self.mode = mode
        self.mask_type = mask_type
        self.pixelwise = pixelwise
        self.box_pro = box_proportion
        self.index_ii = index_ii
        self.index_jj = index_jj
        self.box_size = int(box_proportion * im_shape[-1]) if box_proportion is not None else None
        self.Mask = self.get_mask()
    
    def get_mask(self):
        mask = torch.ones(self.im_shape, device=self.device)
        mask = mask.squeeze()
        if self.mask_type == "random":
            aux = torch.rand_like(mask)
            if self.pixelwise:
                mask[:, aux[0, :, :] > self.mask_rate] = 0
            else:
                mask[aux > self.mask_rate] = 0
        elif self.mask_type == "box" and self.box_size is not None:
            box = torch.zeros(self.box_size, self.box_size).to(self.device)
            if self.index_ii is None and self.index_jj is None:
                self.index_ii = torch.randint(0, mask.shape[-2] - self.box_size , (1,)).item()
                self.index_jj = torch.randint(0, mask.shape[-1] - self.box_size , (1,)).item()
            mask[..., self.index_ii:self.box_size+self.index_ii, self.index_jj:self.box_size+self.index_jj] = box
        else:
            raise ValueError("mask_type not supported")
        return mask.reshape(self.im_shape)
        
    # ---- Precision matrix op ----
    def precision(self, sigma_model, sigma_t, rho):
        return (1 / rho**2) * ( torch.ones_like(self.Mask) - self.Mask / (sigma_model**2 * rho**2 + 1))

    
    # ---- mean vector op ----
    def mean_estimate(self, y, z, sigma_model, sigma_t, rho):
        # ---- Precision  operator ----
        inv_precision_matrix = lambda x: self.precision( sigma_model, sigma_t, rho) * x

        # ---- Solving the linear system Ax = b ----
        temp = self.Mask * y / sigma_model**2 +  z * rho**2
        mean_vector = inv_precision_matrix(temp)
        mean_vector = temp.div(self.Mask / sigma_model**2 + rho**2)
        return mean_vector      

    # ---- Inpainting operator ----
    def Ax(self, x: torch.Tensor)->torch.Tensor:
        x_shape = x.shape
        if len(x_shape) > 2 and len(self.Mask.shape) == 2:
            Mask = self.Mask.repeat(x_shape[-3],1,1)
        else:
            Mask = self.Mask
        return x * Mask
    
    def ATx(self, x:torch.Tensor)->torch.Tensor:
        return self.Ax(x)
    
    def y(self, x:torch.Tensor)->torch.Tensor:
        if not self.transform_y:
            x0 = inverse_image_transform(x)
        else:
            x0 = x
        out = self.forward_operator(x0) + self.sigma_model * torch.randn_like(x0)
        return out
 


    