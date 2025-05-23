import torch
from diffusion.utils import get_data
import deepinv as dinv


dataset = get_data("FFHQ")

im = dataset["test"][0].unsqueeze(0)

centre_mask = torch.ones_like(im)
centre_mask[:, :, 64:3*64, 64:3*64] = 0

phys = dinv.physics.Inpainting(
    tensor_size=im.shape[1:],
    mask = centre_mask, #Centre box
    noise_model = dinv.physics.GaussianNoise(
        sigma=0.05,
    ),
)

betas = torch.linspace(0.0001, 0.02, 1000)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
y = phys(im)
y_adj = phys.A_adj(y)
y_adj = y_adj + torch.nn.ZeroPad2d(64)(torch.randn((1,3,128,128)))
# Load diffusion model
model = dinv.models.DiffUNet()

with torch.no_grad():
    ## DDPM Loop
    im = 2*im - 1
    im_rec = im
    for t in range(100,-1,-10):
        print(f'\r{t}', end="", flush=True)
        im_t = torch.sqrt(alphas_cumprod[t]) * im_rec + torch.sqrt(1 - alphas_cumprod[t]) * torch.randn_like(im)
        # im_t = 0.5 * (im_t + 1)
        sigma = 1-alphas_cumprod[t]
        sigma = sigma.sqrt()
        sigma = sigma * (1+4*sigma**2).sqrt() /2




        # sigma = 1.
        noise_est = model.forward_diffusion(im_t, torch.tensor([t]))
        im_rec = (im_t -noise_est[:,:3,...]*torch.sqrt(1 - alphas_cumprod[t]))/alphas[t].sqrt()
    dinv.utils.plot(
        [im, im_t, im_rec],
        titles=["Original", "Noisy","Denoised"],
    )




