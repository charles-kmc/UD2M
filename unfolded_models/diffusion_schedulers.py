import torch 
import numpy as np
import math

# extract into tensor       
def extract_tensor(
        arr:torch.Tensor, 
        timesteps:torch.Tensor, 
        broadcast_shape:torch.Size
    ) -> torch.Tensor:
        out = arr.to(device=arr.device)[timesteps].float()
        while len(out.shape) < len(broadcast_shape):
            out = out[..., None]
        return out.expand(broadcast_shape)

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
        self.fix_timestep = fix_timestep
        self.num_timesteps = num_timesteps
        if noise_schedule_type == "cosine":
            betas = CosineScheduler(num_timesteps)
        elif noise_schedule_type=="linear":
            betas = LinearScheduler(num_timesteps, beta_start=start_beta, beta_end=end_beta)
        else:
            raise NotImplementedError("This noise schedule is not implemented !!!")
        
        self.betas = torch.from_numpy(betas).to(dtype).to(device)
        self.alphas                  = 1.0 - self.betas
        self.alphas_cumprod          = torch.from_numpy(np.cumprod(self.alphas.cpu().numpy(), axis=0)).to(device)
        self.sqrt_alphas_cumprod     = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod  = torch.sqrt(1. - self.alphas_cumprod)
        self.sigmas   = torch.div(self.sqrt_1m_alphas_cumprod, self.sqrt_alphas_cumprod)
        
    def sample_times(self, bzs, device):
        if self.fix_timestep:
            t_ = torch.randint(1, self.num_timesteps, (1,))
            timestep = (t_ * torch.ones((bzs,))).long().to(device)
        else:
            timestep = torch.randint(1, self.num_timesteps, (bzs,), device=device).long()
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
            extract_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract_tensor(self.sqrt_1m_alphas_cumprod, timesteps, x_start.shape)
            * noise
        )   
    
    #predict x_start from eps 
    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape, ValueError(f"x_t and eps have different shape ({x_t.shape} != {eps.shape})")
        return (
            (x_t - extract_tensor( self.sqrt_1m_alphas_cumprod, t, x_t.shape ) * eps) / 
            extract_tensor(self.sqrt_alphas_cumprod, t, x_t.shape )
        )
        
    # predict eps from x_start
    def predict_eps_from_xstrat(self, x_t, t, xstart):
        assert x_t.shape == xstart.shape, ValueError(f"x_t and eps have different shape ({x_t.shape} != {xstart.shape})")
        return (
            (x_t - extract_tensor(self.sqrt_alphas_cumprod, t, x_t.shape ) * xstart) / 
            extract_tensor( self.sqrt_1m_alphas_cumprod, t, x_t.shape )
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
        ty = self.t_y(sigma_y)
        a1 = sigma_y / (sigma_y + sigma_t)
        a2 = sigma_t / (sigma_y + sigma_t)
        tz = (ty * a1 + t * a2).long()
        rho = extract_tensor(self.scheduler.sigmas, tz, x_shape)
        return rho, tz, ty
    
    # get rho and t from sigma_t and sigma_y
    def get_tz_rho(self, t, sigma_y, x_shape, T_AUG, task="inp"):
        sigma_t = extract_tensor(self.scheduler.sigmas, t, x_shape) 
        if not torch.is_tensor(sigma_y):
            sigma_y = torch.tensor(sigma_y).to(self.device)
        else:
            sigma_y = sigma_y.to(self.device)
        if task == "inp":
            sp = 8
        elif task == "deblur":
            sp = 1
        else:
            sp = 1
        rho = sigma_y * sigma_t * torch.sqrt(1 / (sigma_y**2 + sigma_t**2)) * sp
        tz = torch.clamp(
            self.t_y(rho[0,0,0,0]).cpu() + torch.tensor(T_AUG),
            1,
            self.scheduler.num_timesteps
        )
        tz = (tz * torch.ones((x_shape[0],))).long().to(self.device)
        rho_param = extract_tensor(self.scheduler.sigmas, tz, x_shape)
        ty = self.t_y(sigma_y)
        return rho_param, tz, ty
    
    # getting the corresponding timestep from the diffusion process   
    def t_y(self, sigma_y):
        t_y = torch.clamp(torch.abs(self.scheduler.sigmas.to(self.device) - sigma_y).argmin(), 1, self.scheduler.num_timesteps)
        return t_y    
  