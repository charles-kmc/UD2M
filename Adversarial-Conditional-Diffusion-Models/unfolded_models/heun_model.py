import torch

def extract_tensor(
        arr:torch.Tensor, 
        timesteps:torch.Tensor, 
        broadcast_shape:torch.Size
    ) -> torch.Tensor:
        out = arr.to(device=arr.device)[timesteps].float()
        while len(out.shape) < len(broadcast_shape):
            out = out[..., None]
        return out.expand(broadcast_shape)
    
class HuenModule:
    def __init__(self, device = "cpu", sigma_min=0.002, sigma_max=80, num_timesteps=1000, rho = 7, mode="train"):
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_timesteps = num_timesteps
        self.device = device
        t = torch.linspace(0, self.num_timesteps, self.num_timesteps).to(self.device)
        if mode=="sample":
            self.const1 = self.sigma_max**(1/self.rho) 
            self.const2 = (self.sigma_min**(1/self.rho)-self.sigma_max**(1/self.rho))**self.rho/(self.num_timesteps-1)
            self.sigma_ts = self.const1 + t * self.const2
            self.sigma_prime_ts = torch.ones_like(t)
        elif mode=="train":
            self.sigma_ts = torch.linspace(0.001, 16, self.num_timesteps).to(device)
    
    def sample_xt(self, t, x_start, noise=None):
        sigma_t = extract_tensor(self.sigma_ts, t, x_start.shape)
        if noise is None:
            noise = torch.randn_like(x_start)
        xt = x_start + sigma_t*noise
        
        return xt
    
    def predict_eps_from_xstart(self, t, x_start, x_t):
        sigma_t = extract_tensor(self.sigma_ts, t, x_start.shape)
        eps_pred = (x_t - x_start)/sigma_t
        
        return eps_pred
    
    def predict_xstart_from_eps(self, t, xt, eps):
        sigma_t = extract_tensor(self.sigma_ts, t, self.xt.shape)
        x_pred = xt - sigma_t * eps
        return x_pred
    
    def huen_sampler(self,):
        pass
    