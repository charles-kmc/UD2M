import torch
import numpy as np


class shelving_filters(torch.nn.Module):
    def __init__(self, N = 32, loc = 1/4, mag= 2, device='cpu'):
        super().__init__()
        self.N = N
        self.loc = loc
        self.mag = mag
        self.rng = np.random.default_rng(2)
        self.device = device
    
    def sample_filter(self):
        high_pass = self.rng.uniform() > 0.5
        low_pass = self.rng.uniform() > 0.5
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.filter = torch.ones((self.N, self.N//2+1))

        if high_pass:
            self.high_loc = self.rng.normal(loc=self.loc*self.N//2, scale=4, size=2).clip(0, self.N//2-1)
            print("High loc:", self.high_loc)
            self.filter[int(self.high_loc[0]):-int(self.high_loc[0]), int(self.high_loc[1]):] = 1/self.mag

        if low_pass:
            self.low_loc = self.rng.normal(loc=(1-self.loc)*self.N//2, scale=4, size=2).clip(1, self.N//2-1)
            print("Low loc:", self.low_loc)
            self.filter[:int(self.low_loc[0]), :int(self.low_loc[1])] = self.mag
            self.filter[-int(self.low_loc[0]):, :int(self.low_loc[1])] = self.mag
        
        self.filter = self.filter.to(self.device)
        
    def T_g(self, x, filter = None):
        self.sample_filter()
        if self.high_pass or self.low_pass:
            x_f = torch.fft.rfft2(x)
            x_f = x_f * self.filter
            x = torch.fft.irfft2(x_f)
            self.min = x.min().item()
            self.max = x.max().item()
            x = 2*(x - self.min) / (self.max - self.min) - 1
        return x
    
    def T_g_inv(self, x):
        if self.high_pass or self.low_pass:
            x = (x + 1) * (self.max - self.min) / 2 + self.min
            x_f = torch.fft.rfft2(x)
            x_f = x_f / self.filter
            x = torch.fft.irfft2(x_f)

        return x.clamp(-1,1)