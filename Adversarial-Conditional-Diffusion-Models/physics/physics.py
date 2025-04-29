import random
import os
from PIL import Image
import numpy as np
from collections import defaultdict
from scipy import ndimage
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from .blur_functions import gaussian_kernel, uniform_kernel, uniform_motion_kernel, UniformMotionBlurGenerator
from utils.utils import inverse_image_transform, image_transform
from deepinv.physics.generator import MotionBlurGenerator
import physics.sisr_utils as sisr_utils

import deepinv as dinv


def set_seed(seed):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For GPU operations, set the seed for CUDA as well
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you're using multi-GPU

    # Ensure that deterministic algorithms are used for certain operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
def blur_to_operator(blur, im_size, transpose=False):
       
    if len(blur.shape) > 2:
        raise ValueError("The kernel dimension is not correct")
    device = blur.device
    blur=blur.squeeze()
    if transpose:
        blur = blur.T
    n, m = blur.shape
    H_op = torch.zeros(im_size[-2], im_size[-1], dtype=blur.dtype).to(device)
    H_op[:n,:m] = blur
    for axis, axis_size in enumerate(blur.shape):
        H_op = torch.roll(H_op, -int(axis_size / 2), dims=axis)
    FFT_h = torch.fft.fft2(H_op, dim = (-2,-1))
    # n_ops = torch.sum(torch.tensor(FFT_h.shape) * torch.log2(torch.tensor(FFT_h.shape)))
    # FFT_h[torch.abs(FFT_h)<n_ops*2.22e-6] = torch.tensor(0)
    return FFT_h[None, None, ...]

# physic module  
class Kernels:
    def __init__(
        self,
        operator_name,
        kernel_size,
        device,
        path_motion_kernel:str = "/users/cmk2000/sharedscratch/Datasets/blur-kernels/kernels",
        mode = "test",
        dtype = torch.float32,
    ):
        self.operator_name = operator_name
        self.kernel_size = kernel_size
        self.device = device
        self.mode = mode
        self.dtype = dtype
        
        # motion kernel
        if mode == "train":
            self.path_motion_kernel_train = os.path.join(path_motion_kernel, "training_set")
            self.list_motion_kernel = os.listdir(self.path_motion_kernel_train)
        else:
            self.path_motion_kernel_val = os.path.join(path_motion_kernel, "validation_set")
            self.list_motion_kernel = os.listdir(self.path_motion_kernel_val)
 
    def get_blur(self, seed=None):
        if self.operator_name == "gaussian":
            blur_kernel = self._gaussian_kernel(self.kernel_size).to(self.device)
        elif self.operator_name  == "uniform":
            if seed is not None:
                set_seed(seed)
            blur_kernel = self._uniform_kernel(self.kernel_size).to(self.device)
        elif self.operator_name  == "motion":
            blur_kernel = self._motion_kernel().to(self.device)
        elif self.operator_name == "uniform_motion":
            blur_kernel = self._uniform_motion_kernel(self.kernel_size).to(self.device)
        elif self.operator_name == "bicubic":
            blur_kernel = self._bicubic_kernel(4).to(self.device)  # Fixed to 4 times SR for now

        else:
            raise ValueError(f"Blur type {self.operator_name } not implemented !!")
        return blur_kernel.squeeze()

    def _uniform_motion_kernel(self, kernel_size):
        self.uniform_kernel_generator = MotionBlurGenerator((kernel_size,kernel_size), num_channels=1)#UniformMotionBlurGenerator(kernel_size=(kernel_size,kernel_size))
        if self.mode == "inference":
            torch.random.manual_seed(1675)
        else:
            pass
        return self.uniform_kernel_generator.step(sigma=0.5, l=0.3)["filter"].squeeze()
    
    def _uniform_motion_kernel_new(self, kernel_size):
        generator = UniformMotionBlurGenerator((kernel_size, kernel_size))         
        if self.mode == "inference":
            torch.random.manual_seed(1675)
        else:
            pass
        kernel = generator.step(1)
        return kernel["filter"].squeeze()

    def _bicubic_kernel(self, factor):
        x = np.arange(start=-2 * factor + 0.5, stop=2 * factor, step=1) / factor
        a = -0.5
        x = np.abs(x)
        w = ((a + 2) * np.power(x, 3) - (a + 3) * np.power(x, 2) + 1) * (x <= 1)
        w += (
            (a * np.power(x, 3) - 5 * a * np.power(x, 2) + 8 * a * x - 4 * a)
            * (x > 1)
            * (x < 2)
        )
        w = np.outer(w, w)
        w = w / np.sum(w)
        return torch.Tensor(w)
    
    def _motion_kernel(self):
        # defined transforms 
        transforms_kernel = transforms.Compose([
                transforms.Resize((self.kernel_size, self.kernel_size)),
                transforms.ToTensor()
            ])
        if self.mode == "train":
            kernel_name = random.choice(self.list_motion_kernel) 
            kernel = Image.open(os.path.join(self.path_motion_kernel_train, kernel_name)) 
        else:
            kernel_name = self.list_motion_kernel[7]  
            kernel = Image.open(os.path.join(self.path_motion_kernel_val, kernel_name))
            
        kernel_torch = transforms_kernel(kernel)[0,...].squeeze()
        kernel_torch /= kernel_torch.sum()
        return kernel_torch
    
    def _gaussian_kernel(self, kernel_size, std = 3.0):
        c = int((kernel_size - 1)/2)
        v = torch.arange(-kernel_size + c, kernel_size - c, dtype = self.dtype).reshape(-1,1)
        vT = v.T
        h = torch.exp(-0.5*(v**2 + vT**2)/(2*std))
        eps = np.finfo(np.float32).resolution
        h[h < eps * h.max()] = 0
        h /= torch.sum(h.ravel())
        return h 
        
    def _uniform_kernel(self, kernel_size):
        exp = torch.ones(kernel_size, kernel_size).to(self.dtype)
        exp /= torch.sum(exp)
        return exp  

## ====> Deblurring module       
class Deblurring:
    def __init__(
        self, 
        sigma_model:float= 0.05,
        operator_name:str="gaussian", 
        device:str = "cpu",
        dtype = torch.float32, 
        scale_image:bool = False,
    ) -> None:
        # parameters
        self.operator_name=operator_name
        self.scale_image = scale_image
        self.device = device
        self.sigma_model = 2 * sigma_model if self.scale_image else sigma_model
        self.dtype = dtype
        self.pre_compute = defaultdict(lambda:None)

    def pre_calculate(self, blur, im_shape):
        '''__summary__'''
        FFT_h = self.blur2A(im_shape)
        FFT_hc = torch.conj(FFT_h) if self.operator_name=="gaussian" else self.blur2A(im_shape, transpose=True)
        FFT_hch = torch.pow(torch.abs(FFT_h),2)    
        return {"FFT_h":FFT_h, "FFT_hc":FFT_hc, "FFT_hch":FFT_hch, "FFT_hcy":None}
    
    
    def blur2A(self, im_size, transpose=False):
        if len(self.blur.shape) > 2:
            raise ValueError("The kernel dimension is not correct")
        device = self.blur.device
        blur=self.blur
        if transpose:
            blur = self.blur.T
        n, m = blur.shape
        H_op = torch.zeros(im_size[-2], im_size[-1], dtype=blur.dtype).to(device)
        H_op[:n,:m] = blur
        for axis, axis_size in enumerate(blur.shape):
            H_op = torch.roll(H_op, -int(axis_size / 2), dims=axis)
        FFT_h = torch.fft.fft2(H_op, dim = (-2,-1))
        n_ops = torch.sum(torch.tensor(FFT_h.shape) * torch.log2(torch.tensor(FFT_h.shape)))
        FFT_h[torch.abs(FFT_h)<n_ops*2.22e-6] = torch.tensor(0)
        return FFT_h[None, None, ...]

    # forward operator
    def convolution_xh(self, x, transpose = False):
        device = x.device
        blur = self.blur
        if isinstance(blur, torch.Tensor):
            blur = blur.cpu().squeeze().numpy()
        if transpose:
            blur = blur.T    
        ax = ndimage.convolve(x.cpu().numpy(), blur[None,None,...], mode='wrap')
        return torch.from_numpy(ax).to(device)
        
    def Ax(self, x):
        # im_size = x.shape[-1]
        # FFT_h = self.blur2A(im_size)
        if not hasattr(self, "pre_compute"):
            raise ValueError("pre_compute doesn't exists!!")
        FFT_x = torch.fft.fft2(x, dim = (-2,-1))
        ax = torch.real(
            torch.fft.ifft2(self.pre_compute["FFT_h"] * FFT_x, dim = (-2,-1))
        )
        return ax
    
    # forward operator
    def ATx(self, x):
        # im_size = x.shape[-1]
        # FFT_h = self.blur2A(im_size, transpose=True)
        # FFT_hc = torch.conj(FFT_h)
        if not hasattr(self, "pre_compute"):
            raise ValueError("pre_compute doesn't exists!!")
        FFT_x = torch.fft.fft2(x, dim = (-2,-1))
        atx = torch.real(
            torch.fft.ifft2(self.pre_compute["FFT_hc"] * FFT_x, dim = (-2,-1))
        )
        return atx
    
    # precision
    def _precision(
        self,
        FFT_hch:torch.Tensor, 
        sigma_model:float,
        rho:float, 
        sigma_t:float,
        lambda_:float
    ) -> torch.Tensor:
        out =  FFT_hch / (lambda_*sigma_model**2) + 1 / (sigma_t**2) + 1 / rho**2 
        return 1 / out
    
    # mean estimate
    def mean_estimate(
        self, 
        obs:dict, 
        x:torch.Tensor, 
        params:dict,
    )->torch.Tensor:
        x_t, y = obs["x_t"], obs["y"]
        assert x_t.shape == x.shape == y.shape, "x_t,y and x should have the same shape"
        if not hasattr(self, "pre_compute"):
            raise ValueError("pre_compute doesn't exists!!")        
        if self.operator_name=="gaussian":
            FFT_hcy = self.pre_compute["FFT_hcy"]      
            FFT_hch = self.pre_compute["FFT_hch"]
            FFTx = torch.fft.fft2(x, dim=(-2,-1))
            FFTx_t = torch.fft.fft2(x_t, dim=(-2,-1))
            
            temp = FFT_hcy / (params["lambda_sig"]*params["sigma_model"]**2) + FFTx / params["rho"]**2 + FFTx_t  / (params["sigma_t"]**2 * params["sqrt_alpha_comprod"])
            prec_temp = self._precision(FFT_hch, params["sigma_model"], params["rho"], params["sigma_t"], params["lambda_sig"])
            zest = torch.real(torch.fft.ifft2(temp * prec_temp, dim = (-2,-1)))
        else:
            zest = self.gradient_based_update(obs, x,params)
        return zest
    
    def gradient_based_update(self, obs, x:torch.Tensor, params:dict):
        ds_max_iter=params["ds_max_iter"] if params["ds_max_iter"] is not None else 5
        z = x.clone()
        for ii in range(ds_max_iter):
            g1 = self.convolution_xh(self.convolution_xh(z)-obs["y"], transpose=True) / (params["lambda_sig"]*params["sigma_model"]**2)
            g2 = (z - x)/params["rho"]**2
            g3 = (z-obs["x_t"]/params["sqrt_alpha_comprod"])/params["sigma_t"]**2
            grad_z = g1 / params["lambda_sig"] + g2 + g3
            delta =1/torch.norm(g1+g2+g3)
            z = z - delta * grad_z
        return z
    
    # observation y
    def y(self, x:torch.Tensor, blur:torch.Tensor)->torch.Tensor:
        self.blur = blur
        im_size = x.shape
        FFT_h = self.blur2A(im_size)
        self.pre_compute["FFT_h"] = FFT_h # = self.pre_calculate(blur, x.shape)
        if not self.scale_image:
            x0 = inverse_image_transform(x)
        else:
            x0 = x
        if self.operator_name != "gaussian":
            Ax0 = self.convolution_xh(x0)
        else:
            Ax0 = self.Ax(x0)
        y_ = Ax0 + self.sigma_model * torch.randn_like(Ax0)
        
        FFT_hc = torch.conj(FFT_h) if self.operator_name=="gaussian" else self.blur2A(im_size, transpose=True)
        if self.operator_name !="gaussian":
            ATy = self.convolution_xh(y_, transpose=True)
            FFT_hcy = torch.fft.fft2(ATy,dim=(-2,-1))
        else:
            FFT_y = torch.fft.fft2(y_, dim=(-2, -1))
            FFT_hcy = FFT_y*FFT_hc
        
        # pre computes
        self.pre_compute["FFT_hc"] = FFT_hc
        self.pre_compute["FFT_hch"] = FFT_hc*FFT_h
        self.pre_compute["FFT_hcy"] = FFT_hcy
        return y_

class Inpainting:
    def __init__(
        self, 
        mask_rate: float,
        im_size:int,
        operator_name:str = "random",
        device:str = "cpu",
        sigma_model:float= 0.05, 
        box_proportion = 0.4, 
        index_ii = None, 
        index_jj = None,
        dtype = torch.float32, 
        scale_image:bool = False,
        mode = "test",
    ) -> None:
        # parameters
        self.im_size = im_size
        self.scale_image = scale_image
        self.device = device
        self.sigma_model = 2 * sigma_model if self.scale_image else sigma_model
        self.dtype = dtype
        self.mask_rate = mask_rate
        self.mode = mode
        self.operator_name = operator_name
        self.box_pro = box_proportion if box_proportion is not None else 0.4
        self.box_size = int(box_proportion * im_size)
        
        self.Mask = self.get_mask()
    
    def get_mask(self, index_ii=None, index_jj=None):
        mask = torch.ones((self.im_size,self.im_size), device=self.device)
        mask = mask.squeeze()
        if self.operator_name == "random":
            aux = torch.rand_like(mask)
            mask[aux > self.mask_rate] = 0
        elif self.operator_name == "box" and self.box_size is not None:
            box = torch.zeros(self.box_size, self.box_size).to(self.device)
            if index_ii is None or index_jj is None:
                if self.mode!="inference":
                    #set_seed(3456789)
                    index_ii = torch.randint(0, mask.shape[-2] - self.box_size , (1,)).item()
                    index_jj = torch.randint(0, mask.shape[-1] - self.box_size , (1,)).item()
                if self.mode=="inference":
                    index_ii,index_jj = 48,34
            print(index_ii, index_jj, self.box_size)
            mask[..., index_ii:self.box_size+index_ii, index_jj:self.box_size+index_jj] = box
        else:
            raise ValueError("operator_name not supported")
        return mask.reshape(self.im_size, self.im_size)
        
    # ---- Precision matrix op ----
    def precision(
        self, 
        sigma_model:float, 
        sigma_t:torch.Tensor, 
        rho:torch.Tensor,
        lambda_:float
    )->torch.Tensor:
        # cf. Sherman-Morrison-Woodbury formula
        para_temp1 = (rho**2 * sigma_t**2 ) / (sigma_t**2  + rho**2)
        para_temp2 = lambda_*sigma_model**2 if sigma_model!=0 else 1
        prec_temp2 = para_temp1 * (torch.ones_like(self.Mask) - para_temp1 * self.Mask / (para_temp2 + para_temp1))
        # prec_temp2 = 1 / (self.Mask / para_temp2 +  1 / rho**2 +  1 / sigma_t**2)
        return  prec_temp2

    # ---- mean vector op ----
    def mean_estimate(self, 
        obs:dict, 
        x:torch.Tensor, 
        params:dict,
    )->torch.Tensor:
        x_t = obs["x_t"]
        y = obs["y"]
        assert x_t.shape == x.shape == y.shape, "x_t, y and x should have the same shape"
        # ---- Precision  operator ----
        inv_precision_matrix = lambda x: self.precision(params["sigma_model"], params["sigma_t"], params["rho"], params["lambda_sig"]) * x
        para_temp = params["lambda_sig"]*params["sigma_model"]**2 if params["sigma_model"]!=0 else 1
        
        # ---- Solving the linear system Ax = b ----
        temp = self.Mask * y / para_temp +  x / params["rho"]**2 +  x_t / (params["sigma_t"]**2 * params["sqrt_alpha_comprod"]) 
        zest = inv_precision_matrix(temp)
        # zest[:,:,self.Mask.to(torch.bool)] = y[:,:,self.Mask.to(torch.bool)]#+0.4*zest[:,:,self.Mask.to(torch.bool)] 
        return zest      

    # ---- Inpainting operator ----
    def Ax(self, x:torch.Tensor)->torch.Tensor:
        Mask = self.Mask
        return x * Mask
    
    def ATx(self, x:torch.Tensor)->torch.Tensor:
        return self.Ax(x)
    
    def y(self, x:torch.Tensor)->torch.Tensor:
        if not self.scale_image:
            x0 = inverse_image_transform(x)
        else:
            x0 = x
        x0_ = x0 + self.sigma_model * torch.randn_like(x0)
        out = self.Ax(x0_)
        return out
    
## ====> Superresolution module
class SuperResolution:
    def __init__(
        self, 
        im_size:int,
        device:str = "cpu",
        sigma_model:float= 0.05, 
        dtype = torch.float32, 
        scale_image:bool = False,
        mode = "test",
    ) -> None:
        # parameters
        self.im_size = im_size
        self.scale_image = scale_image
        self.device = device
        self.sigma_model = 2 * sigma_model if self.scale_image else sigma_model
        self.dtype = dtype
        self.mode = mode
        self.bool = 1
    
    def upsample(self, x):
        '''s-fold upsampler
        Upsampling the spatial size by filling the new entries with zeros
        x: tensor image, NxCxWxH
        '''
        sf = self.sf
        z = F.interpolate(x, size=(x.shape[-2]*sf, x.shape[-1]*sf), mode='bilinear') 
        return z

    def downsample(self, x):
        sf = self.sf
        st = 0
        return x[..., st::sf, st::sf]
    
    def y(self, x:torch.Tensor, blur:torch.Tensor, sf:int)->torch.Tensor:
        '''__summary__
        Generate observation.
        
        inputs:
            - x (tensor): high-resolution image
            - blur (tensor): blur kernel
            - sf (int): resolution factor
        outputs:
            - obs (tensor): low-resolution image
        '''
        self.sf = sf
        self.blur = blur
        if self.scale_image:
            x0 = inverse_image_transform(x)
        else:
            x0 = x 
        
        # observation model     
        FFT_h = blur_to_operator(blur, x0.shape)
        FFT_x = torch.fft.fft2(x0, dim = (-2,-1))
        Hx = torch.real(torch.fft.ifft2(FFT_h * FFT_x, dim = (-2,-1)))
        Lx = self.downsample(Hx)
        
        if self.scale_image:
            Lx = image_transform(Lx)
        
        obs =  Lx + self.sigma_model * torch.randn_like(Lx)
        
        # pre compute
        FFT_hc = torch.conj(FFT_h)
        if self.scale_image and self.bool:
            obs_ = (obs+1)/2
        else:
            obs_=obs.clone()
           
        FFT_y = torch.fft.fft2(obs_, dim = (-2,-1))
        FFT_hcy = FFT_hc*torch.fft.fft2(self.upsample(obs_), dim=(-2, -1))
        self.pre_compute = {"FFT_h":FFT_h, "FFT_hch":FFT_h**2, "FFT_hc":FFT_hc, "FFT_y":FFT_y, "FFT_hcy":FFT_hcy}
        
        return obs
    
    def mean_estimate00(
        self,
        obs:dict, 
        x:torch.Tensor, 
        params:dict
    ):
        ## 
        if self.bool:
            x = (x+1)/2
            xt = (obs["x_t"]+1)/2
            params["sigma_model"] = params["sigma_model"]/2
        else:
            xt = obs["x_t"]
        if not hasattr(self, "pre_compute"):
            raise ValueError("pre_compute is None")
        sf = self.sf
        
        ## 
        var = params["lambda_sig"]*params["sigma_model"]**2
        alpha_z = 1/(params["rho"]**2)
        alpha_xt = 1/(params["sigma_t"]**2)
        alpha = alpha_z + alpha_xt
        alpha_var = (alpha*var).ravel()[0]
        
        # cf. Woodbury formular
        F_rhs = self.pre_compute["FFT_hcy"] + torch.fft.fft2(alpha_z*x, dim=(-2,-1)) + torch.fft.fft2((alpha_xt/params["sqrt_alpha_comprod"])*xt, dim=(-2,-1)) 
        z1 = self.pre_compute["FFT_h"].mul(F_rhs)
        
        FBR = torch.mean(sisr_utils.splits(z1, sf), dim=-1, keepdim=False)
        invW = torch.mean(sisr_utils.splits(self.pre_compute["FFT_hch"], sf), dim=-1, keepdim=False)
        invWBR = FBR.div(invW+alpha_var)
        FCBinvWBR = self.pre_compute["FFT_hc"]*invWBR.repeat(1, 1, sf, sf)
        
        Fz = (F_rhs - FCBinvWBR)/alpha
        zest = torch.real(torch.fft.ifft2(Fz, dim=(-2, -1)))
        
        # scale from [0,1] to [-1,1]
        if self.scale_image and self.bool:
            zest = torch.clamp(zest,0,1)
            zest = image_transform(zest)
        return zest
    
    def mean_estimate00(
        self,
        obs:dict, 
        x:torch.Tensor, 
        params:dict,
    ):
        ## TODO
        x_t = obs["x_t"]
        if not hasattr(self, "pre_compute"):
            raise ValueError("pre_compute is None")
        ## TODO
        sigma_y = 1/(params["lambda_sig"]*params["sigma_model"]**2)
        alpha_z = 1/params["rho"]**2
        alpha_xt = 1/(params["sigma_t"]**2)
        alpha = alpha_z + alpha_xt

        F_rhs = self.pre_compute["FFT_hcy"]*sigma_y + alpha_z*torch.fft.fft2(x, dim=(-2,-1)) + (alpha_xt/params["sqrt_alpha_comprod"])*torch.fft.fft2(x_t, dim=(-2,-1)) 
        co_variance = self.pre_compute["FFT_hch"]*sigma_y + alpha
        Fz = F_rhs/co_variance
        
        zest = torch.real(torch.fft.ifft2(Fz, dim=(-2, -1)))
        return zest
 
    def mean_estimate(
        self,
        obs:dict, 
        x:torch.Tensor, 
        params:dict,
        ):   
             
        # ---> Compute right-hand vector in frequency domain
        if self.bool:
            x = (x+1)/2
            xt = (obs["x_t"]+1)/2
            sigma_model = params["sigma_model"]/2
        else:
            xt = obs["x_t"]
            sigma_model = params["sigma_model"]
        x_fft = torch.fft.fft2(x, dim = (-2,-1)) 
        xt_fft = torch.fft.fft2(xt, dim = (-2,-1)) 
        assert x_fft.shape == xt_fft.shape == self.pre_compute["FFT_hcy"].shape, f"x ({x_fft.shape}) and xt ({xt_fft.shape}) should have the same shape!!"
        
        right_fft = self.pre_compute["FFT_hcy"] / (params["lambda_sig"]*sigma_model**2) + x_fft/(params["rho"]**2) + xt_fft / (params["sigma_t"]**2*params["sqrt_alpha_comprod"])

        # ---> Covariance in fft
        denom_fft = self.pre_compute["FFT_hch"]/(params["lambda_sig"]*sigma_model**2) + 1/params["rho"]**2 + 1/params["sigma_t"]**2
        
        # ---> Solve in the frequency domain
        zest_fft = right_fft / denom_fft

        # ---> Inverse FFT to get back to the spatial domain
        zest = torch.real(torch.fft.ifft2(zest_fft, dim = (-2,-1)))
        
        # ---> scale from [0,1] to [-1,1]
        if self.scale_image and self.bool:
            zest = torch.clamp(zest,0,1)
            zest = image_transform(zest)
        return zest
    
class CT():
    def __init__(self, 
                im_size:int,
                n_channels:int=3,
                angles = 78,
                device:str = "cpu",
                sigma_model:float= 0.05, 
                dtype = torch.float32, 
                scale_image:bool=False,
                circle:bool=False,
                mode = "test"
            ):
        self.bool =1
        self.angles = angles
        self.mode = mode
        self.n_channels = n_channels
        self.dtype = dtype
        self.scale_image = scale_image
        self.device = device
        self.sigma_model = sigma_model
        self.op = dinv.physics.Tomography(
                            img_width=im_size,
                            angles=angles,
                            circle=circle,
                            device=device,
                            noise_model=dinv.physics.GaussianNoise(sigma=sigma_model),
                        )
        PI = 4 * torch.ones(1).atan()
        self.SCALING = (PI / (2 * self.angles)).to(self.device)  
    
    def init(self, y):
        return torch.clamp(self.ATx(y).add(1).div(2),0.,1.) * self.SCALING
    
    def y(self, x:torch.Tensor)->torch.Tensor:
        if self.scale_image and self.bool:
            x0 = torch.clamp(x.add(1).div(2),0,1)
        else:
            x0 = x.clone()
        y_ = self.op(x0)
        if self.scale_image and self.bool:
            y_ = y_.mul(2).add(-1)
        return y_
    
    def ATx(self, y:torch.Tensor)->torch.Tensor:
        val = self.op.A_adjoint(y)
        # if self.scale_image:
        #     return val.mul(2).add(-1)
        return val
    
    def Ax(self, x:torch.Tensor)->torch.Tensor:
        # if self.scale_image:
        # x = torch.clamp(x.add(1).div(2),0., 1.)
        val = self.op.A(x)
        return val
    
    def grad(self, x:torch.Tensor, z:torch.Tensor, obs:dict, params:dict)->dict:

        grad1=self.op.A_adjoint(self.op(x)-obs["y"])
        grad2=x-z
        grad3=x-obs["x_t"]/params["sqrt_alpha_comprod"]
        return {"grad1":grad1, "grad2":grad2, "grad3":grad3}
        
    def mean_estimate(
        self,
        obs:dict, 
        x:torch.Tensor, 
        z:torch.Tensor, 
        params:dict,
        ):   
    
        # ---> Compute right-hand vector in frequency domain
        if self.bool:
            x = (x+1)/2
            z = (z+1)/2
            obs["x_t"] = obs["x_t"].add(1).div(2)
            sigma_model = params["sigma_model"]/2
        else:
            sigma_model = params["sigma_model"]
           
        step_size_algo = params["step_algo"] * self.SCALING
        
        for _ in range(params["max_iter_algo"]):
            grads = self.grad(z, x, obs, params)
            z = z - step_size_algo * (grads["grad1"]/(params["lambda_sig"]*sigma_model**2) + grads["grad2"]/params["rho"]**2 + grads["grad3"]/params["sigma_t"]**2 )
        
        # ---> scale from [0,1] to [-1,1]
        if self.scale_image and self.bool:
            zest = torch.clamp(z,0.,1.)
            zest = image_transform(zest)
        else:
            zest = z.clone()
        return zest