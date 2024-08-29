import torch 
import deepinv as dinv
from models.unfolded.unrolledPhysics import UnfoldedBlurFFT
from .train_weights import get_id_batch, get_im_tuple
from math import prod
from torchvision.transforms import Grayscale, ToTensor, Resize, InterpolationMode

from utils.data import FFHQ

def train_weights_and_denoiser(
        physics,
        W_init,
        denoiser,
        img_shape,
        train_data,
        device,
        n_iter=4000, 
        log_every_n=1000,
        lr = 1e-2,
        batch_size = 128,
        n_test=128,
        n_stop = 5,
        verbose = 1,
        lda = 1.,
        sigma=0.05
):
    log_W_mat = W_init.log()
    log_W_mat.requires_grad = True
    dim = prod(img_shape)
    optimizer = torch.optim.Adam([{"params":[log_W_mat]},{"params":denoiser.parameters(),"lr":1e-4}], lr=lr)
    template = "{0:9}: {1:8} | {2:4}: {3:20} | {4:4}: {5:20} | {6:4}: {7:20}"
    l_mean = "NA"
    psnrs = []
    with torch.enable_grad():
        epochs = n_iter // log_every_n
        for e in range(epochs):
            with torch.no_grad():
                x = torch.stack([get_im_tuple(train_data[i]) for i in range(n_test)]).to(device)
                y = physics.A(x)
                x_hat = physics.A_adjoint_unrolled(y, weight_mask=log_W_mat.exp())
                psnr = dinv.utils.cal_psnr(x,x_hat)
                psnrs.append(psnr)
                if e > n_stop: 
                    # Catch to stop overfitting
                    if sum([psnrs[-i-1] < psnrs[-i-2]for i in range(n_stop)]) == n_stop:  
                        print("Early stopping")
                        break
            if verbose >=1:
                print(template.format(
                        "Iteration", e*log_every_n,
                        "Loss", l_mean,
                        "Mean", log_W_mat.exp().mean(),
                        "PSNR", psnr
                    ),flush=True)
            l_mean = 0
            pen_mean = 0
            for n in range(log_every_n):
                optimizer.zero_grad()
                ## Batch
                basis = get_id_batch(batch_size, img_shape, device)
                ## Reconstruction
                reconstruction = physics.A_adjoint_unrolled(physics.A(basis), weight_mask=log_W_mat.exp())
                ## Loss 
                loss = dim*(basis - reconstruction).square().sum() / batch_size 
                ## Penalty
                noise = physics.V(physics.V_adjoint(torch.randn_like(basis)*sigma) * log_W_mat.exp().sqrt())
                pen = lda * denoiser(noise, sigma).square().sum()/batch_size
                loss += pen
                ## Optimization step 
                loss.backward()
                optimizer.step()
                l_mean += loss
                pen_mean += pen
                if verbose == 2:
                    print(template.format(
                        "Iteration", n +1 + e*log_every_n,
                        "Loss", l_mean/(n+1),
                        "Mean", log_W_mat.exp().mean(),
                        "PSNR", "----"
                    ) + f"   pen: {pen_mean/(n+1)}", end = '\r', flush=True)
            l_mean /= log_every_n
        return log_W_mat.detach().exp(), denoiser

