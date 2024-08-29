import deepinv as dinv
from torch.nn import Module
from torch import fft
from torch import tensor
import torch
from math import log
from deepinv.physics import DecomposablePhysics, Physics
import numpy as np
from torchvision import transforms
from unfolded_models import unrolledPhysics
from unfolded_models.DiffUNet import DiffUNet



## Function to load data
def get_dataset(datasetname="FFHQ", img_width=None):
    if img_width is not None:
        train_transforms = (
            transforms.Resize((img_width,img_width), interpolation=transforms.InterpolationMode.BICUBIC), 
            transforms.ToTensor()
        )
        test_transforms = (
            transforms.Resize((img_width,img_width), interpolation=transforms.InterpolationMode.BICUBIC), 
            transforms.ToTensor()
        )
    else:
        train_transforms = tuple([
            transforms.ToTensor()
        ])
        test_transforms = tuple([
            transforms.ToTensor()
        ])
    if datasetname == "FFHQ":
        ## Using FFHQ dataset
        from unfolded_utils.data import FFHQ
        train_set = FFHQ(train=True,transforms=train_transforms)
        test_set = FFHQ(indices=[i for i in range(69400,69495)] + [69037,69133,69367,69887,69929],transforms=test_transforms)
    elif datasetname == "MNIST":
        ## USING MNIST dataset
        from torchvision.datasets import MNIST
        train_set = MNIST(
            root = './data/MNIST',
            download = True,
            train=True,
            transform=transforms.Compose(train_transforms)
        )
        test_set = MNIST(
            root = './data/MNIST',
            download = True,
            train=False,
            transform=transforms.Compose(test_transforms)
        )
    elif datasetname == "Div2K":
        ## USING MNIST dataset
        from deepinv.datasets import DIV2K
        train_set = DIV2K(
            root = './data/Div2K',
            download = True,
            mode='train',
            transform=transforms.Compose(train_transforms)
        )
        test_set = DIV2K(
            root = './data/Div2K',
            download = True,
            mode='val',
            transform=transforms.Compose(test_transforms)
        )
    img_shape = train_set[0].shape if not isinstance(train_set[0], tuple) else train_set[0][0].shape
    return train_set, test_set, img_shape

## Function to determine forward operator
def get_physics_operator(
        noise_model             =               "gaussian",
        degradation             =               "gaussian_deblurring",
        gaussian_noise_level    =               0.01,
        gaussian_blur_sd        =               (3,3),
        img_size                =               (3, 32, 32),
        kernel_size             =               (7,7),
        device                  =               "cpu",
        rhomat                  =               False,
        scaling_domain          =               "adjoint",
        unfolded_physics        =               False
    ):
    physics_generator=None
    ## Define noise operator
    if noise_model == "gaussian":
        noise = dinv.physics.GaussianNoise(
            sigma=gaussian_noise_level
        )  

        data_fidelity = dinv.optim.data_fidelity.L2(
            sigma = gaussian_noise_level
        )
    else: 
        raise NotImplementedError(
            "Noise model {} is not implemented".format(noise_model)
        )
    
    ## Define degradation operator
    if "deblurring" in degradation:
        if degradation == "gaussian_deblurring":
            blur_kernel = dinv.physics.blur.gaussian_blur(
                sigma=gaussian_blur_sd,
                angle=0
            )
        
            if blur_kernel.shape[-2] >= kernel_size[-2] and blur_kernel.shape[-1] >= kernel_size[-1]:
                blur_kernel = blur_kernel[
                    ...,
                    blur_kernel.shape[-2]//2 - kernel_size[0]//2:blur_kernel.shape[-2]//2 + kernel_size[0]//2+1,
                    blur_kernel.shape[-1]//2 - kernel_size[1]//2:blur_kernel.shape[-1]//2 + kernel_size[1]//2+1
                ]
                blur_kernel /= blur_kernel.sum()
        elif "motion" in degradation and "rand" not in degradation:
            try:
                knum = int(degradation[-2:])
            except:
                print("WARNING: unspecified kernel number, using default motion blur")
                knum=0
            blur_kernel = unrolledPhysics.get_motion_blur(knum)
            ## Load motion blur
        elif "rand_motion_deblurring" in degradation:
            blur_kernel = torch.from_numpy(unrolledPhysics.Kernel(size=kernel_size, intensity=0.5, rng=np.random.default_rng(1)).kernelMatrix).unsqueeze(0).unsqueeze(0)
            if "sample" in degradation:
                physics_generator = unrolledPhysics.UniformMotionBlurGenerator(kernel_size=kernel_size)
        physics = unrolledPhysics.rhomatBlurFFT(
            img_size=img_size,
            filter=blur_kernel,
            device=device,
            noise_model=noise,
            real_fft=(not rhomat) and (rhomat=="adjoint"),
            scaling_domain=scaling_domain
        )

    elif "inpainting" in degradation:
        physics = dinv.physics.Inpainting(
            tensor_size=img_size,
            mask=0.5,
            pixelwise=True,
            device=device,
            noise_model=noise
        )
    
    elif "superres" in degradation:
        physics = dinv.physics.Downsampling(
            img_size=img_size,
            filter="bicubic",
            factor = int(degradation.partition("X")[-1]) if "X" in degradation else 4,
            device=device,
            padding='circular',
            noise_model=noise
        )

    else:
        raise NotImplementedError(
            "Degradation operator {} is not implemented".format(degradation)
        )
    return physics, data_fidelity, physics_generator


def get_prior(
    prior           =       "pnp",
    denoiser        =       "DnCNN",
    channels        =       3,
    width           =       256,
    pretrained      =       None,
    train           =       False,
    device          =       "cpu",
    equivariant     =       True,
    ckpt_den        =       False,
    x_train         =       None,
    x_test          =       None,
    latent_dim      =       100,
    batch_norm       =       True,
    ):
    if prior == "pnp" or prior == "pnp_score":
        if denoiser == "DnCNN":
            denoiser = dinv.models.DnCNN(
                in_channels=channels,
                out_channels=channels,
                pretrained=pretrained,
                train=train,
                device=device
            )
        elif denoiser == "GSDRUNet":
            denoiser = dinv.models.GSDRUNet(
                in_channels=3,
                pretrained="download",
                device=device
            )
        elif denoiser == "DRUNet":
            denoiser = dinv.models.DRUNet(
                in_channels=channels,
                out_channels=channels,
                pretrained='download',
                train=False,
                device=device
            )
        elif denoiser == "UNet":
            denoiser = dinv.models.UNet(
                in_channels=channels,
                out_channels=channels,
                scales=4 if width >= 128 else 2,
                batch_norm = batch_norm,
            ).to(device)
            if train:
                denoiser.train()
        elif denoiser == "DiffUNet10m":
            denoiser = DiffUNet(
                device=device,
                use_lora=train,
                model_path='./'
            )
        else:
            raise NotImplementedError("Denosier {} is undefined".format(denoiser))
        
        if equivariant:
            denoiser = dinv.models.EquivariantDenoiser(denoiser,transform='rotations')
        if ckpt_den:
            denoiser = unrolledPhysics.ckpt_denoiser(denoiser)
        prior_model = (
            dinv.optim.PnP(denoiser = denoiser)
            if prior == "pnp" else 
            dinv.optim.ScorePrior(denoiser=denoiser)
        )
            
    elif "L1" in prior:
        wv = prior[2:] if len(prior) > 2 else "db8"
        prior_model = dinv.optim.WaveletPrior(
            level=3,
            wv=wv,
            p=1,
            device=device,
            wvdim=2,
        )
    else:
        raise NotImplementedError("Prior {} is undefined".format(prior))
    
    for p in prior_model.parameters():
            p.requires_grad=train
    return prior_model 

