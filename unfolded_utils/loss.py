import deepinv as dinv 
import torch 



class SW(dinv.loss.loss.Loss):
    def __init__(self, num_projections = 1000, device = None):
        super(SW, self).__init__()
        self.num_projections = num_projections
        self.device=device
        self.name="SW"

    
    def forward(self, x_net, x, y, physics, model, **kwargs):
        x_net = torch.flatten(x_net, start_dim=1)
        x = torch.flatten(x, start_dim=1)

        ## Sample random projections
        proj = torch.randn((x.shape[-1], self.num_projections), device=self.device)
        proj = proj / proj.square().sum(dim=0).sqrt()
        
        ## Project data 
        x_net = x_net.mm(proj).sort(dim=0)[0]
        x = x.mm(proj).sort(dim=0)[0]

        loss = (x_net - x).square().mean()

        return loss
