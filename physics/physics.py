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
from deepinv.physics.blur import bicubic_filter
import deepinv as dinv


def set_seed(seed):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you're using multi-GPU
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
        # path_motion_kernel:str = "/users/cmk2000/sharedscratch/Datasets/blur-kernels/kernels",
        mode = "test",
        dtype = torch.float32,
        std = 3.0,
        sf = 4,
    ):
        self.operator_name = operator_name
        self.kernel_size = kernel_size
        self.device = device
        self.mode = mode
        self.dtype = dtype
        self.std = std
        self.sf = sf
        
    def get_blur(self, seed=None):
        if self.operator_name == "gaussian":
            blur_kernel = self._gaussian_kernel(self.kernel_size, std = self.std).to(self.device)
        elif self.operator_name  == "uniform":
            if seed is not None:
                set_seed(seed)
            blur_kernel = self._uniform_kernel(self.kernel_size).to(self.device)
        elif self.operator_name == "anisotropic":
            blur_kernel = self._anisotropic_kernel(self.kernel_size).to(self.device)
        elif self.operator_name  == "motion":
            blur_kernel = self._motion_kernel(self.kernel_size).to(self.device)
        elif self.operator_name == "uniform_motion":
            blur_kernel = self._uniform_motion_kernel().to(self.device)
        elif self.operator_name == "bicubic":
            blur_kernel = bicubic_filter(self.sf).to(self.device)
        else:
            raise ValueError(f"Blur type {self.operator_name } not implemented !!")
        return blur_kernel.squeeze()
    
    def _anisotropic_kernel(self, kernel_size):
        sigma = 10  # better make argument for kernel type
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(
            self.device)
        kernel = torch.matmul(kernel[:,None],kernel[None,:])/torch.sum(kernel)**2
        return kernel

    def _motion_kernel(self, kernel_size):
        self.motion_kernel_generator = MotionBlurGenerator((kernel_size,kernel_size), num_channels=1)#UniformMotionBlurGenerator(kernel_size=(kernel_size,kernel_size))
        if self.mode == "inference":
            torch.random.manual_seed(1675)
        else:
            pass
        return self.motion_kernel_generator.step(sigma=0.5, l=0.1)["filter"].squeeze()

    def _motion_kernel_new(self, kernel_size):
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
    
    def _uniform_motion_kernel(self):
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
    
    def _gaussian_kernel(self, kernel_size, std = 10.0):
        c = int((kernel_size - 1)/2)
        v = torch.arange(-kernel_size + c+1, kernel_size - c, dtype = self.dtype).reshape(-1,1)
        vT = v.T
        h = torch.exp(-0.5*(v**2 + vT**2)/(std))
        eps = np.finfo(np.float32).resolution
        h[h < eps * h.max()] = 0
        h /= torch.sum(h.ravel())
        return h 
    
    def _gaussian_kernel_(self, kernel_size, std = 10.0):
        c = int((kernel_size - 1)/2)
        v = torch.arange(-kernel_size + c+1, kernel_size - c, dtype = self.dtype).reshape(-1,1)
        func = lambda x: torch.exp(-0.5*(x**2)/(std))
        val = torch.stack([func(a) for a in v])
        h = torch.zeros(kernel_size, kernel_size)
        h[c,:] = val.reshape(-1)
        h /= torch.sum(h.ravel())
        return h
        
    def _uniform_kernel(self, kernel_size):
        exp = torch.ones(kernel_size, kernel_size).to(self.dtype)
        exp /= torch.sum(exp)
        return exp  


__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper

def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)

## ====> Deblurring module 
@register_operator(name="deblur")        
class Deblurring:
    def __init__(
        self, 
        sigma_model:float= 0.05,
        operator_name:str="gaussian", 
        device:str = "cpu",
        dtype = torch.float32, 
        scale_image:bool = False,
        *args,
        **kwargs
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
            x = inverse_image_transform(x)
      
        Ax = self.Ax(x)
        y_ = Ax + self.sigma_model * torch.randn_like(Ax)

        FFT_hc = torch.conj(FFT_h)
        FFT_y = torch.fft.fft2(y_, dim=(-2, -1))
        FFT_hcy = FFT_y*FFT_hc
        
        # pre computes
        self.pre_compute["FFT_hc"] = FFT_hc
        self.pre_compute["FFT_hch"] = FFT_hc*FFT_h
        self.pre_compute["FFT_hcy"] = FFT_hcy
        return y_

@register_operator(name="inp")
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
        *args,
        **kwargs
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
        
        self.Mask = self.get_mask(index_ii=index_ii, index_jj=index_jj)
        
    def get_mask(self, index_ii=None, index_jj=None):
        mask = torch.ones((self.im_size,self.im_size), device=self.device)
        mask = mask.squeeze()
        if self.operator_name == "random":
            aux = torch.rand_like(mask)
            mask[aux > self.mask_rate] = 0
        elif self.operator_name == "box" and self.box_size is not None:
            box = torch.zeros(self.box_size, self.box_size).to(self.device)
            if index_ii is None or index_jj is None:
                if self.mode=="inference":
                    index_ii,index_jj = 64,64 #48,34
                else:
                    index_ii = torch.randint(0, mask.shape[-2] - self.box_size , (1,)).item()
                    index_jj = torch.randint(0, mask.shape[-1] - self.box_size , (1,)).item()
            
            index_ii,index_jj = 64,64        
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
        para_temp = params["lambda_sig"]*params["sigma_model"]**2 
        
        # ---- Solving the linear system Ax = b ----
        temp = self.Mask * y / para_temp +  x / params["rho"]**2 +  x_t / (params["sigma_t"]**2 * params["sqrt_alpha_comprod"]) 
        zest = inv_precision_matrix(temp)
        return zest      

    # ---- Inpainting operator ----
    def A(self, x:torch.Tensor)->torch.Tensor:
        Mask = self.Mask
        return x * Mask
    
    def AT(self, x:torch.Tensor)->torch.Tensor:
        return self.A(x)
    
    def y(self, x:torch.Tensor)->torch.Tensor:
        x = torch.clamp(x.add(1).div(2), 0, 1)
        y = x + self.sigma_model/2 * torch.randn_like(x)
        y = self.A(y)
        y = image_transform(y)
        return y

## ====> Superresolution module
@register_operator(name="sr")
class SuperResolution:
    def __init__(
        self, 
        im_size:int,
        device:str = "cpu",
        sigma_model:float= 0.05, 
        dtype = torch.float32, 
        scale_image:bool = False,
        mode = "test",
        *args,
        **kwargs
    ) -> None:
        # parameters
        self.im_size = im_size
        self.scale_image = scale_image
        self.device = device
        self.sigma_model = 2*sigma_model if self.scale_image else sigma_model
        self.dtype = dtype
        self.mode = mode
    
    def upsample(self, x):
        sf = self.sf
        z = F.interpolate(x, size=(x.shape[-2]*sf, x.shape[-1]*sf), mode='bilinear') 
        return z

    def downsample(self, x):
        sf = self.sf
        st = 0
        return x[..., st::sf, st::sf]
    
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
    
    def A(self, x):
        if not hasattr(self, "pre_compute"):
            raise ValueError("pre_compute doesn't exists!!")
        FFT_x = torch.fft.fft2(x, dim = (-2,-1))
        ax = torch.real(
            torch.fft.ifft2(self.pre_compute["FFT_h"] * FFT_x, dim = (-2,-1))
        )
        return ax
    
    # forward operator
    def AT(self, x):
        if not hasattr(self, "pre_compute"):
            raise ValueError("pre_compute doesn't exists!!")
        FFT_x = torch.fft.fft2(x, dim = (-2,-1))
        atx = torch.real(
            torch.fft.ifft2(self.pre_compute["FFT_hc"] * FFT_x, dim = (-2,-1))
        )
        return atx
 
    def y(self, x:torch.Tensor, blur:torch.Tensor, sf:int)->torch.Tensor:
        self.pre_compute = {}
        self.sf, self.blur, im_size = sf, blur, x.shape
        if not self.scale_image:
            x = inverse_image_transform(x)
        # observation model   
        FFT_h = self.blur2A(im_size)  
        self.pre_compute["FFT_h"]=FFT_h
        Hx = self.A(x)
        Lx = self.downsample(Hx)
        y_ =  Lx + self.sigma_model * torch.randn_like(Lx)
                
        # pre compute
        FFT_hc = torch.conj(FFT_h)
        FFT_hcy = FFT_hc*torch.fft.fft2(image_transform(self.upsample(inverse_image_transform(y_))), dim=(-2, -1))
        self.pre_compute["FFT_hc"] = FFT_hc
        self.pre_compute["FFT_hch"] = FFT_h**2
        self.pre_compute["FFT_hcy"] = FFT_hcy
             
        return y_
 
    def mean_estimate(
        self,
        obs:dict, 
        x:torch.Tensor, 
        params:dict,
        *args,
        **kwargs
        ) -> torch.Tensor:   
        x_t, sigma_model = obs["x_t"], params["sigma_model"]
        x_fft, xt_fft = torch.fft.fft2(x, dim = (-2,-1)) , torch.fft.fft2(x_t, dim = (-2,-1)) 
        
        assert x_fft.shape == xt_fft.shape == self.pre_compute["FFT_hcy"].shape, f"x ({x_fft.shape}) and xt ({xt_fft.shape}) should have the same shape!!"
        right_fft = self.pre_compute["FFT_hcy"]/(params["lambda_sig"]*sigma_model**2) + x_fft/(params["rho"]**2) + xt_fft/(params["sigma_t"]**2*params["sqrt_alpha_comprod"])

        # ---> Covariance in fft
        denom_fft = self.pre_compute["FFT_hch"]/(params["lambda_sig"]*sigma_model**2) + 1/params["rho"]**2 + 1/params["sigma_t"]**2
        zest_fft = right_fft/denom_fft
        zest = torch.real(torch.fft.ifft2(zest_fft, dim = (-2,-1)))
 
        return zest

class JPEG:
    def __init__(self, 
            im_size:int, 
            device:str = "cpu", 
            sigma_model:float= 0.05, 
            dtype = torch.float32, 
            scale_image:bool=False,
            qf:int=10,
            *wargs,
            **kwargs
        )-> None:
        self.im_size = im_size
        self.device = device
        self.dtype = dtype
        self.scale_image = scale_image
        self.qf = qf
        self.jpeg_model=build_jpeg(qf)
        self._set_sigma(sigma_model)
        self.noise = get_noise(name="gaussian", sigma=self.sigma_model)

    def _set_sigma(self, sigma_model: float) -> None:
        self.sigma_model = 2*sigma_model if self.scale_image else sigma_model
        # self.sigma_model = sigma_model

    def init(self, y: torch.Tensor) -> torch.Tensor:
        return self.jpeg_model(y)

    def y(self, x:torch.Tensor) -> torch.Tensor: 
        return self.noise(self.jpeg_model(x))

    def A(self, x:torch.Tensor) -> torch.Tensor:
        return encode_jpeg(x, self.qf)

    def AT(self, x:torch.Tensor) -> torch.Tensor:
        return self.jpeg_model(x, self.qf)
    
    def grad_likelihood(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        # return self.jpeg_model(self.jpeg_model(x) - y)
        return self.jpeg_model(x) - y

    def mean_estimate_GD(
        self, 
        z:torch.Tensor, 
        y:torch.Tensor, 
        x_t:torch.Tensor, 
        params, 
        x_init:torch.Tensor=None,
        *wargs, 
        **kwargs
    )-> torch.Tensor:
        # gradients
        grad_0=lambda x: self.grad_likelihood(x, y)/(params["sigma_model"]**2*params["lambda_sig"])
        grad_1=lambda x: (x-x_t/params["sqrt_alpha_comprod"])/params["sigma_t"]**2
        grad_2=lambda x: (x-z)/params["rho"]**2
        grad = lambda x: (grad_0(x) + grad_1(x) + grad_2(x))
        # initialisation
        if x_init is None:
            x = self.init(y)
        else: 
            x = x_init.clone()
        
        # GD loop
        for it in range(params["max_iter_algo"]):
            g0, g1, g2 = grad_0(x), grad_1(x), grad_2(x)
            grad = (g0+g1+g2)/torch.norm(g0+g1+g2)
            x-= params["step_algo"]*grad  
        return x
    
    def loss(self, x, y):
        l = self.jpeg_model(x) - y
        return torch.norm(l, p=2)**2

    def mean_estimate(
        self, 
        z:torch.Tensor, 
        y:torch.Tensor, 
        x_t:torch.Tensor, 
        params, 
        x_init:torch.Tensor=None,
        *wargs, 
        **kwargs
    )-> torch.Tensor:
        # gradients
        grad_1=lambda x: (x-x_t/params["sqrt_alpha_comprod"])/params["sigma_t"]**2
        grad_2=lambda x: (x-z)/params["rho"]**2
        grad = lambda x: grad_1(x) + grad_2(x)
        # initialisation
        if x_init is None:
            x = self.init(y)
        else: 
            x = x_init.clone()
        
        # ADAM loop
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        m, v = torch.zeros_like(x), torch.zeros_like(x)
        # for it in range(params["max_iter_algo"]):
        for it in range(3):
            with torch.enable_grad():
                x = x.requires_grad_(True)
                l = self.loss(x, y)
                g = torch.autograd.grad(l, x, retain_graph=True)[0]
            
            g = g / (params["sigma_model"]**2 * params["lambda_sig"])            
            
            with torch.no_grad():
                g = grad(x) + g
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                m_hat = m / (1 - beta1 ** (it + 1))
                v_hat = v / (1 - beta2 ** (it + 1))
                x -= params["step_algo"] * m_hat / (torch.sqrt(v_hat) + eps)
            
        return x    
