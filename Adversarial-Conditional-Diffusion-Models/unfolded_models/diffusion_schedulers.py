import torch 
import numpy as np
import math
import models as models
     

def LinearScheduler(timesteps, beta_start = 0.0001, beta_end=0.02):
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end
    betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
    return betas

def CosineScheduler(timesteps, beta_max =0.999):
    cos_function = lambda x: math.cos((x + 0.008) / (1.008) * math.pi / 2)**2
    betas = []
    for t in range(timesteps):
        t1 = t/timesteps
        t2 = (t+1)/timesteps
        beta = min(1 - cos_function(t2) / cos_function(t1), beta_max)
        betas.append(beta)
    return np.array(betas)

# diffusion scheduler module        
class DiffusionScheduler:
    def __init__(
        self, 
        start_beta:float = 0.0001, 
        end_beta: float = 0.02, 
        num_timesteps:int = 1000,
        dtype = torch.float32,
        device = "cpu", 
        fix_timestep = True,
        noise_schedule_type:str = "linear"
    ) -> None:
        self.device = device
        self.fix_timestep = fix_timestep
        self.num_timesteps = num_timesteps
        if noise_schedule_type == "cosine":
            betas = CosineScheduler(num_timesteps)
        elif noise_schedule_type=="linear":
            betas = LinearScheduler(num_timesteps, beta_start=start_beta, beta_end=end_beta)
        else:
            raise NotImplementedError("This noise schedule is not implemented !!!")
        
        self.betas = torch.from_numpy(betas).to(dtype).to(self.device)
        self.alphas                  = 1.0 - self.betas
        self.alphas_cumprod          = torch.from_numpy(np.cumprod(self.alphas.cpu().numpy(), axis=0)).to(self.device)
        self.sqrt_alphas_cumprod     = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod  = torch.sqrt(1. - self.alphas_cumprod)
        self.sigmas   = torch.div(self.sqrt_1m_alphas_cumprod, self.sqrt_alphas_cumprod)
        self.deg_cont = 3
        self.sqrt_alphas_cumprod_cont = torch.from_numpy(
            np.polyfit(np.arange(0, num_timesteps), self.sqrt_alphas_cumprod.cpu().log().numpy(), self.deg_cont),
        ).to(self.device)
        self.inv_sigmas_cont = torch.from_numpy(
            np.polyfit(self.sigmas.cpu().log().numpy(), np.arange(0, num_timesteps), self.deg_cont),
        ).to(self.device)

        # print(f"alpha_cont_diff: {torch.from_numpy(np.polyval(self.sqrt_alphas_cumprod_cont, np.arange(0, num_timesteps))).exp() - self.sqrt_alphas_cumprod.cpu()}")
        # exit()
        
    def sample_times(self, bzs):
        if self.fix_timestep:
            t_ = torch.randint(1, self.num_timesteps, (1,))
            timestep = (t_ * torch.ones((bzs,))).long().to(self.device)
        else:
            timestep = torch.randint(1, self.num_timesteps, (bzs,), device=self.device).long()
        return timestep
    
    def sample_xt(
        self, 
        x_start:torch.Tensor, 
        timesteps:torch.Tensor, 
        noise:torch.Tensor
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            models.extract_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + models.extract_tensor(self.sqrt_1m_alphas_cumprod, timesteps, x_start.shape)
            * noise
        )   

    def alpha_cont_diff(self, t):  ## Interpolate alphas for continuous time approximation
        """
        alpha_cont_diff: \alpha_t = \alpha_0 + \alpha_1 t + \alpha_2 t^2 + \alpha_3 t^3
        """
        return sum([
            self.sqrt_alphas_cumprod_cont[-i-1] * t**i for i in range(self.deg_cont)
        ]).exp()
    
    def inv_sigmas_cont_diff(self, sig):
        """
        inv_sigmas_cont_diff: \sigma_t = \sigma_0 + \sigma_1 t + \sigma_2 t^2 + \sigma_3 t^3
        """
        return sum([
            self.inv_sigmas_cont[-i-1] * sig**i for i in range(self.deg_cont)
        ])

    def sample_xt_cont(
        self, 
        x_start:torch.Tensor, 
        timesteps:torch.Tensor, 
        noise:torch.Tensor
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        sqrt_alp_cont = self.alpha_cont_diff(timesteps)
        while len(sqrt_alp_cont.shape) < len(x_start.shape):
            sqrt_alp_cont = sqrt_alp_cont[..., None]
        return (
            sqrt_alp_cont * x_start
            + (1-sqrt_alp_cont.square()).sqrt()
            * noise
        )   
    
    #predict x_start from eps 
    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape, ValueError(f"x_t and eps have different shape ({x_t.shape} != {eps.shape})")
        return (
            (x_t - models.extract_tensor( self.sqrt_1m_alphas_cumprod, t, x_t.shape ) * eps) / 
            models.extract_tensor(self.sqrt_alphas_cumprod, t, x_t.shape )
        )
    
    #predict x_start from eps 
    def predict_xstart_from_eps_cont(self, x_t, t, eps):
        assert x_t.shape == eps.shape, ValueError(f"x_t and eps have different shape ({x_t.shape} != {eps.shape})")
        sqrt_alp_cont = self.alpha_cont_diff(t)
        while len(sqrt_alp_cont.shape) < len(x_t.shape):
            sqrt_alp_cont = sqrt_alp_cont[..., None]
        return (
            (x_t - (1 - sqrt_alp_cont.square()).sqrt() * eps) / 
            sqrt_alp_cont
        )
        
    # predict eps from x_start
    def predict_eps_from_xstrat(self, x_t, t, xstart):
        assert x_t.shape == xstart.shape, ValueError(f"x_t and eps have different shape ({x_t.shape} != {xstart.shape})")
        return (
            (x_t - models.extract_tensor(self.sqrt_alphas_cumprod, t, x_t.shape ) * xstart) / 
            models.extract_tensor( self.sqrt_1m_alphas_cumprod, t, x_t.shape )
        )
    
    # get sequence of timesteps 
    def get_seq_progress_seq(self, iter_num):
        seq = np.sqrt(np.linspace(0, self.num_timesteps**2, iter_num))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1
        progress_seq = seq[::max(len(seq)//10,1)]
        if progress_seq[-1] != seq[-1]:
            progress_seq.append(seq[-1])
        seq = seq[::-1]
        progress_seq = progress_seq[::-1]
        return (seq, progress_seq)

# coustomise timesteps for the denoising step  
class GetDenoisingTimestep:
    def __init__(self, device):
        self.scheduler = DiffusionScheduler(device=device)
        self.device = device
    
    # getting differently the timestep   
    def get_tz_ty_rho(self, t, sigma_y, x_shape):
        print(f"sigma y {sigma_y}\n\n")
        sigma_t = self.scheduler.sigmas.to(self.device)[t]
        if not torch.is_tensor(sigma_y):
            sigma_y = torch.tensor(sigma_y).to(self.device)
        else:
            sigma_y = sigma_y.to(self.device)
        ty = self._noise_step(sigma_y)
        a1 = sigma_y / (sigma_y + sigma_t)
        a2 = sigma_t / (sigma_y + sigma_t)
        tz = (ty * a1 + t * a2).long()
        rho = models.extract_tensor(self.scheduler.sigmas, tz, x_shape)
        return rho, tz, ty
    
    # get rho and t from sigma_t and sigma_y
    def get_z_timesteps(self, t, sigma_y, x_shape, T_AUG, sp, **kwargs):
        sigma_t = models.extract_tensor(self.scheduler.sigmas, t, x_shape) 
        if not torch.is_tensor(sigma_y):
            sigma_y = torch.tensor(sigma_y).to(self.device)
        else:
            sigma_y = sigma_y.to(self.device)
        
        
        rho_unscaled = sigma_y * sigma_t * torch.sqrt(1 / (sigma_y**2 + sigma_t**2))  ## For finding T
        rho =  rho_unscaled* sp   ### For HQS
        
        tz = torch.clamp(
            self._noise_step(rho_unscaled.ravel()[0]).cpu() + T_AUG,
            1,
            self.scheduler.num_timesteps
        )
        tz = (tz * torch.ones((x_shape[0],), device=t.device))
        # rho_param = models.extract_tensor(self.scheduler.sigmas, tz, x_shape)
        ty = self._noise_step(sigma_y)
        return rho, tz, ty
    
    # getting the corresponding timestep from the diffusion process   
    def _noise_step(self, sigma):
        t_op = torch.clamp(torch.abs(self.scheduler.sigmas.to(self.device) - sigma).argmin(), 1, self.scheduler.num_timesteps)
        return t_op    
  