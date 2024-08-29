import torch 
import deepinv as dinv
from math import prod

def train_W(
        physics,
        W_init,
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
        lda = 1e-1,
        sigma=0.05
):
    log_W_mat = W_init.log()
    log_W_mat.requires_grad = True
    dim = prod(img_shape)
    optimizer = torch.optim.Adam([log_W_mat], lr=lr)
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
            for n in range(log_every_n):
                optimizer.zero_grad()
                ## Batch
                basis = get_id_batch(batch_size, img_shape, device)
                ## Reconstruction
                reconstruction = physics.A_adjoint_unrolled(physics.A(basis), weight_mask=log_W_mat.exp())
                ## Loss 
                loss = dim*(basis - reconstruction).square().sum() / batch_size 
                ## Penalty
                pen = physics.A_adjoint_unrolled(torch.randn_like(basis)*sigma, weight_mask = log_W_mat.exp())
                loss += lda*pen.square().sum()
                ## Optimization step 
                loss.backward()
                optimizer.step()
                l_mean += loss
                if verbose == 2:
                    print(template.format(
                        "Iteration", n +1 + e*log_every_n,
                        "Loss", l_mean/(n+1),
                        "Mean", log_W_mat.exp().mean(),
                        "PSNR", "----"
                    ), end = '\r', flush=True)
            l_mean /= log_every_n
        return log_W_mat.detach().exp()

def rand_basis(img_shape):
    k = [
        torch.randint(0,img_shape[0],[1]),
        torch.randint(0,img_shape[1],[1]),
        torch.randint(0,img_shape[2],[1]),
    ]
    e_k = torch.zeros((1,*img_shape))
    e_k[(0,*k)] = 1
    return e_k

def get_im_tuple(im):
    if isinstance(im, tuple) or isinstance(im, list):
        return im[0]
    else: 
        return im

def get_id_batch(N, img_shape, device):
    return torch.cat([rand_basis(img_shape) for i in range(N)], dim=0).to(device)