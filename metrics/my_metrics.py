import deepinv as dinv 
import torch 
import numpy as np


class Coverage:
    def __init__(self, embedding = None, num_alphas = 100, device=None):
        self.embedding = embedding
        if embedding is None: 
            from .vae import VAEfc
            from torchvision.transforms import CenterCrop
            vae = VAEfc(
                x_shape=(28, 28), latent_dim=12, enc_hidden_dims=[512, 512], d_activ=torch.nn.functional.relu, return_latent_sd=False
            ).to(device)
            transforms = torch.nn.Sequential(CenterCrop(28), torch.nn.Flatten())
            vae.load_state_dict(torch.load('/users/js3006/unrolled_cgan/models/saved/MNIST_VAE.pth.tar'))

            def embedding(x):
                x = (x - x.min()) / (x.max() - x.min()+1e-8)
                x_t = transforms(x)
                return vae.encoder(x_t)
            self.embedding = embedding
        self.num_alphas = num_alphas
        self.reset()
    
    def reset(self):
        self.alphas = np.linspace(0,1,self.num_alphas)
        self.sums = np.zeros(self.num_alphas)
        self.sums_meanemb = np.zeros(self.num_alphas)
        self.samples = 0
    
    def update(self, x_samples, x):
        """
            x_samples: (N, S, C, H, W) tensor of generated samples
            x: (N, C, H, W) tensor of reference samples
        """
        if self.embedding:
            x_meanemb = self.embedding(x_samples.mean(dim=1)).detach().cpu().numpy()
            x_samples = np.stack([self.embedding(x_t).detach().cpu().numpy() for x_t in x_samples])   
            x = self.embedding(x).detach().cpu().numpy()
        else:
            x_samples = x_samples.detach().cpu().numpy().reshape((x_samples.shape[0], -1))
            x = x.detach().cpu().numpy().reshape((x.shape[0], -1))

        x_mean = np.mean(x_samples, axis=1)
        dist_mean = np.linalg.norm(x - x_mean, axis=-1)
        dist_meanemb = np.linalg.norm(x_meanemb - x, axis=-1)
        dist_samples = np.linalg.norm(x_samples - x_mean[:,None,:], axis=-1)
        dist_samples = np.sort(dist_samples, axis=1)
        dist_samples_meanemb = np.linalg.norm(x_samples - x_meanemb[:,None,:], axis=-1)
        dist_samples_meanemb = np.sort(dist_samples_meanemb, axis=1)
        for i, alpha in enumerate(self.alphas):
            self.sums[i] += np.sum(dist_mean <= dist_samples[:, int(alpha*(dist_samples.shape[1]-1))])
            self.sums_meanemb[i] += np.sum(dist_meanemb <= dist_samples_meanemb[:, int(alpha*(dist_samples_meanemb.shape[1]-1))])
        self.samples += x.shape[0]
    
    def compute(self, csv_path = None):
        """
            Compute coverage metric
        """
        cov =  self.sums / self.samples
        cov_meanemb = self.sums_meanemb / self.samples
        if csv_path:
            out = {"alpha": self.alphas, "coverage": cov, "coverage_meanemb": cov_meanemb}
            from pandas import DataFrame
            df = DataFrame(out)
            df.to_csv(csv_path, index=False)
        return cov

class SW(dinv.loss.Loss):
    def __init__(self, num_projections = 1000, device = None, x_loader=None, physics = None):
        super(SW, self).__init__()
        self.num_projections = num_projections
        self.device=device
        self.name="SW"
        self.x_loader = x_loader  # Random dataloader for x
        if x_loader:
            if physics:
                self._len_load = len(self.x_loader)
                self._iter_x = iter(self.x_loader)
                self._n_done = 0
                self.sw_base_est = self.estimate_sw(physics)
                print(f"Base SW Estimate: {self.sw_base_est}")
            self._len_load = len(self.x_loader)
            self._iter_x = iter(self.x_loader)
            self._n_done = 0
    
    def estimate_sw(self, physics, n_mc = 32):
        mcest = 0
        for i in range(n_mc):
            x1 = next(iter(self.x_loader))
            if isinstance(x1, tuple) or isinstance(x1, list):
                x1 = x1[0]
            x1 = x1.to(self.device)
            y1 = physics(x1)
            loss = self.forward(x1, None, y1, physics, None)
            mcest += loss
        return mcest / n_mc
    
    def gen_x(self, physics):
        if self._n_done >= self._len_load - 1:
            self._iter_x = iter(self.x_loader)
            self._n_done = 0
        self._n_done += 1
        x_samples = next(self._iter_x)
        if isinstance(x_samples, tuple) or isinstance(x_samples, list):
            x_samples = x_samples[0]
        x_samples = x_samples.to(self.device)
        y_samples = physics(x_samples)
        return torch.cat([x_samples,y_samples], dim=-1)
    
    def forward(self, x_net, x, y, physics, model, **kwargs):
        if self.x_loader: # If comparing to random samples of joint distribution
            x = self.gen_x(physics)
            x_net = torch.cat([x_net, y], dim=-1)
        x_net = torch.flatten(x_net, start_dim=1)
        x = torch.flatten(x, start_dim=1)

        # ## Sample random projections
        # proj = torch.randn((x.shape[-1], self.num_projections), device=self.device)
        # proj = proj / proj.square().sum(dim=0).sqrt()
        
        # ## Project data 
        # x_net = x_net.mm(proj).sort(dim=0)[0]
        # x = x.mm(proj).sort(dim=0)[0]

        # loss = (x_net - x).square().mean()
        loss = calculate_sw_from_ims(x_net, x, num_projections=self.num_projections, device=self.device)


        return loss

class SW_metric_test:
    """
        SW metric to be used during testing
    """
    def __init__(self, img_size, num_projections = 10000, device = None, seed = 3489723):
        self.num_projections = num_projections
        self.device=device 
        self.img_size = img_size
        self.rng = np.random.default_rng(seed)
        if img_size*num_projections < 1e9:
            self.proj = self.rng.standard_normal((img_size, num_projections))
            self.proj = self.proj / np.linalg.norm(self.proj, axis=0)
            self.proj = torch.from_numpy(self.proj).float().to(device)
        else:
            raise ValueError("Too many projections for memory")
        self._reset()
        
    
    def _reset(self):
        self.x_proj = []
        self.x_net_proj = []

    def add_projections(self, x_net = None, x = None):
        """
            True images x and fake images x_net. Can be called iteratively.
        """
        if x_net is not None:
            x_net = x_net.flatten(start_dim=1)
            x_net = x_net @ self.proj
            self.x_net_proj.append(x_net.cpu())
        if x is not None:
            x = x.flatten(start_dim=1)
            x = x @ self.proj
            self.x_proj.append(x.cpu())
    

    def calculate_sw(self):
        """
            Compute the SW metric from the stored projections.
            Requires add_projections to be called for both x and x_net
        """
        # Ensure same number of projections for each set
        if len(self.x_net_proj) > len(self.x_proj):
            _x_net_proj = self.x_net_proj[:len(self.x_proj)]
            _x_proj = self.x_proj
        elif len(self.x_proj) > len(self.x_net_proj):
            _x_proj = self.x_proj[:len(self.x_net_proj)]
            _x_net_proj = self.x_net_proj
        else: 
            _x_proj = self.x_proj
            _x_net_proj = self.x_net_proj
        if len(_x_net_proj) == 0:
            return None
        x_net = torch.cat(_x_net_proj, dim=0).sort(dim=0)[0]
        x = torch.cat(_x_proj, dim=0).sort(dim=0)[0]
        loss = (x_net - x).square().mean()
        return loss
    
    @property
    def SW(self):
        """
            Property to calculate SW metric
        """
        return self.calculate_sw()

class CFID_metric_test:
    def __init__(self, embedding_network, embedding_dim, device=None):
        self.embedding_network = embedding_network
        self.embedding_dim = embedding_dim
        self.device=device
        self._reset()  # Initialize all mean and variance matrices to zero
        
    def _reset(self):
        self.m_x_true = torch.zeros(self.embedding_dim, device=self.device)
        self.m_y_true = torch.zeros(self.embedding_dim, device=self.device)
        self.m_y_predict = torch.zeros(self.embedding_dim, device=self.device)
        self.C_xx = torch.zeros((self.embedding_dim, self.embedding_dim), device=self.device)
        self.C_yy = torch.zeros((self.embedding_dim, self.embedding_dim), device=self.device)
        self.C_xhxh = torch.zeros((self.embedding_dim, self.embedding_dim), device=self.device)
        self.C_xy = torch.zeros((self.embedding_dim, self.embedding_dim), device=self.device)
        self.C_xhy = torch.zeros((self.embedding_dim, self.embedding_dim), device=self.device)
        self.tot_samples = 0
        self.stats = {}

    def _sample_batch(self, x_net, x, y):
        # Calculate embeddings 
        if x_net.ndim == 5:
            x_net_emb = torch.cat([self.embedding_network(x_net[:,i,...]) for i in range(x_net.shape[1])], dim=0)
        else:
            x_net_emb = self.embedding_network(x_net)
        x_emb = self.embedding_network(x)
        y_emb = self.embedding_network(y)
        if x_net.ndim == 5:
            x_emb = x_emb.repeat(x_net.shape[1], 1)
            y_emb = y_emb.repeat(x_net.shape[1], 1)
        new_samples = x_emb.shape[0]

        # Update means
        self.m_x_true = (self.m_x_true*self.tot_samples + x_emb.sum(dim=0)) / (self.tot_samples + new_samples)
        self.m_y_true = (self.m_y_true*self.tot_samples + y_emb.sum(dim=0)) / (self.tot_samples + new_samples)
        self.m_y_predict = (self.m_y_predict*self.tot_samples + x_net_emb.sum(dim=0)) / (self.tot_samples + new_samples)

        # Update covariance matrices
        self.C_xx = (self.C_xx*self.tot_samples + x_emb.t() @ x_emb) / (self.tot_samples + new_samples)
        self.C_yy = (self.C_yy*self.tot_samples + y_emb.t() @ y_emb) / (self.tot_samples + new_samples)
        self.C_xhxh = (self.C_xhxh*self.tot_samples + x_net_emb.t() @ x_net_emb) / (self.tot_samples + new_samples)
        self.C_xy = (self.C_xy*self.tot_samples + x_emb.t() @ y_emb) / (self.tot_samples + new_samples)
        self.C_xhy = (self.C_xhy*self.tot_samples + x_net_emb.t() @ y_emb) / (self.tot_samples + new_samples)

        inv_cyy = torch.linalg.pinv(self.C_yy)

        # Calculate CFID
        C_x_cond_y = self.C_xx - torch.matmul(self.C_xy, torch.matmul(inv_cyy, self.C_xy.t()))
        C_xh_cond_y = self.C_xhxh - torch.matmul(self.C_xhy, torch.matmul(inv_cyy, self.C_xhy.t()))
        C_xy_m_xhy = self.C_xy - self.C_xhy
        tr_1 = torch.trace(torch.matmul(C_xy_m_xhy, torch.matmul(inv_cyy, C_xy_m_xhy.t())))
        tr_2 = torch.trace(C_x_cond_y + C_xh_cond_y) - 2.*trace_sqrt_product_torch(C_x_cond_y, C_xh_cond_y)
        m_dist = torch.norm(self.m_x_true - self.m_y_predict)**2
        C_dist = tr_1 + tr_2
        cfid = m_dist + C_dist 
        self.tot_samples += new_samples

        fid = m_dist + torch.trace(self.C_xx + self.C_xhxh) - 2.*trace_sqrt_product_torch(self.C_xx, self.C_xhxh)
        self.stats = {"FID": fid.detach().cpu(), "m_dist": m_dist.detach().cpu(), "c_dist": C_dist.detach().cpu(), "CFID":cfid.detach().cpu()}
        # Calculate FID 
        return self.stats
    
    def __call__(self, x_npy, x_ref_npy, y_npy):
        """
            Calculate CFID between x_npy and x_ref_npy conditioned on y_npy
            x_npy: (N, C, H, W) or (N, T, C, H, W) tensor of generated images
            x_ref_npy: (N, C, H, W) tensor of reference images
            y_npy: (N, C, H, W) tensor of conditioning images
        """
        self._reset()
        batch_size = 32
        n_samples = x_ref_npy.shape[0]
        for i in tqdm(range(0, n_samples, batch_size), desc="Calculating CFID"):
            x_net = torch.from_numpy(x_npy[i:i+batch_size]).float().to(self.device)
            x = torch.from_numpy(x_ref_npy[i:i+batch_size]).float().to(self.device)
            y = torch.from_numpy(y_npy[i:i+batch_size]).float().to(self.device)
            self._sample_batch(x_net, x, y)
        return self.stats

def calculate_sw_from_ims(x_net, x, num_projections = 1000, device = None):
    x_net = torch.flatten(x_net, start_dim=1)
    x = torch.flatten(x, start_dim=1)

    ## Sample random projections
    proj = torch.randn((x.shape[-1], num_projections), device=device)
    proj = proj / proj.square().sum(dim=0).sqrt()
    
    ## Project data 
    x_net = x_net.mm(proj).sort(dim=0)[0]
    x = x.mm(proj).sort(dim=0)[0]

    loss = (x_net - x).square().mean()

    return loss

# The following code is adapted from https://github.com/matt-bendel/rcGAN/blob/master/evaluation_scripts/cfid/cfid_metric.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

import numpy as np

import torchvision.transforms as transforms
# from utils.mri.math import tensor_to_complex_np
# from utils.mri.transforms import unnormalize_complex
from tqdm import tqdm

def symmetric_matrix_square_root_torch(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    u, s, v = torch.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = s
    si[torch.where(si >= eps)] = torch.sqrt(si[torch.where(si >= eps)])

    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return torch.matmul(torch.matmul(u, torch.diag(si)), v)


def trace_sqrt_product_torch(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.
    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
      => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
      => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
      => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                    = sum(sqrt(eigenvalues(A B B A)))
                                    = sum(eigenvalues(sqrt(A B B A)))
                                    = trace(sqrt(A B B A))
                                    = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.
    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma
    Returns:
      The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = symmetric_matrix_square_root_torch(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))

    return torch.trace(symmetric_matrix_square_root_torch(sqrt_a_sigmav_a))



######## CMMD metric ########
import clip 
import torch
from torchvision.transforms import ToPILImage
import numpy as np
_SIGMA = 10.0
_SCALE = 1000.
class CMMDMetric:
    def __init__(self, model_name="ViT-L/14@336px", sigma = 10.):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.num_ims = 0
        self.sum = 0
        self.sum_ims_cond = 0
        self.num_ims_cond = 0
        self.sigma = sigma
        self.PILim = ToPILImage()
    
    def kernel(self, x, y):
        # Compute the kernel between two sets of features
        kernel_matrix = torch.exp(-((x-y).square().sum(dim=-1)) / (self.sigma ** 2)).clip(0, None)
        return kernel_matrix
    

    def MMD(self, x_enc, xhat_enc):
        # Compute the MMD between two sets of features
        x_md = torch.mean(
            torch.stack(
                [
                    self.kernel(x_enc, x_enc.roll(shifts=(i+1,0), dims=(0,1))) for i in range(len(x_enc)-1)
                ]
            )
        )
        xhat_md = torch.mean(
            torch.stack(
                [
                    self.kernel(xhat_enc, xhat_enc.roll(shifts=(i+1, 0),dims = (0,1))) for i in range(len(xhat_enc)-1)
                ]
            )
        )
        xy_md = torch.mean(
            torch.stack(
                [
                    self.kernel(x_enc, xhat_enc.roll(shifts=(i, 0), dims=(0,1))) for i in range(len(xhat_enc))
                ]
            )
        )
        return x_md + xhat_md - 2 * xy_md

    def CMMD(self, x, y):
        """Memory-efficient MMD implementation in JAX. Modified from https://github.com/sayakpaul/cmmd-pytorch

            This implements the minimum-variance/biased version of the estimator described
            in Eq.(5) of
            https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
            As described in Lemma 6's proof in that paper, the unbiased estimate and the
            minimum-variance estimate for MMD are almost identical.

            Note that the first invocation of this function will be considerably slow due
            to JAX JIT compilation.

            Args:
            x: The first set of embeddings of shape (n, embedding_dim).
            y: The second set of embeddings of shape (n, embedding_dim).

            Returns:
            The MMD distance between x and y embedding sets.
        """
        print("\nCalculating CMMD score... using {} embeddings".format(x.shape[0]))
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        x_sqnorms = torch.diag(torch.matmul(x, x.T))
        y_sqnorms = torch.diag(torch.matmul(y, y.T))

        gamma = 1 / (2 * _SIGMA**2)
        k_xx = torch.mean(
            torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
        )
        k_xy = torch.mean(
            torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
        )
        k_yy = torch.mean(
            torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
        )

        return _SCALE * (k_xx + k_yy - 2 * k_xy)
            
    def _processing(self, x):
        # Preprocess image for CLIP model
        PILx = [
            self.PILim(x[i].cpu()) for i in range(len(x))
        ]
        x_proc = torch.stack(
            [
                self.preprocess(PILx[i]).to(self.device) for i in range(len(PILx))
            ]
        )
        return x_proc
    
    def reset(self):
        # Reset the metric
        self.num_ims = 0
        self.sum = 0
        self.sum_ims_cond = 0
        self.num_ims_cond = 0
        self.x_emb = []
        self.xhat_emb = []
        self.y_emb = []

    def compute(self, x, xhat, y = None):
        # Resize + Normalize images for loaded CLIP model
        # Option to condition images on y to learn joint distribution
        x = (x - x.min()) / (x.max() - x.min()+1e-7)
        xhat = (xhat - x.min()) / (x.max() - x.min()+1e-7)
        xhat = xhat.clip(0,1)
        x_proc = self._processing(x)
        xhat_proc = self._processing(xhat)
        
        
        # Compute CLIP embedding
        x_features = self.model.encode_image(x_proc)
        xhat_features = self.model.encode_image(xhat_proc)
        self.x_emb.append(x_features.cpu().numpy())
        self.xhat_emb.append(xhat_features.cpu().numpy())
        if y is not None:
            y_proc = self._processing(y)
            y_features = self.model.encode_image(y_proc)
            self.y_emb.append(y_features.cpu().numpy())
        
    
    @property
    def mean(self):
        # Return mean CMMD score - Redundant, kept for compatibility
        # CMMD of embedded distributions
        x_emb_cat = np.concatenate(self.x_emb, axis=0)
        xhat_emb_cat = np.concatenate(self.xhat_emb, axis=0)
        
        self.cmmd_score = self.CMMD(
            x_emb_cat,
            xhat_emb_cat
        )
        return self.cmmd_score
    
    @property
    def mean_cond(self):
        if len(self.y_emb) > 0:
            xy_cond = np.concatenate((
                np.concatenate(self.x_emb, axis=0), 
                np.concatenate(self.y_emb, axis=0)
            ), axis=-1)
            xhaty_cond = np.concatenate((
                np.concatenate(self.xhat_emb, axis=0), 
                np.concatenate(self.y_emb, axis=0)
            ), axis=-1)
            self.cond_cmmd_score = self.CMMD(
                xy_cond,
                xhaty_cond
            )
        else: 
            self.cond_cmmd_score = 0
        # Return mean conditional CMMD score - Redundant, kept for compatibility
        return self.cond_cmmd_score
