import torch 

# Function to initialize hqs given the observation 
# obs = (y, x_t)
def init_hqs_diff(obs, physics):
    return {"est":(obs[-1], obs[-1])}

class HQSDiff(torch.nn.Module):
    """
        Implements a single HQS step for the diffusion model.
    """
    def __init__(self):
        super(HQSDiff, self).__init__()
        self.has_cost = False ## Silences warning
        self.F_fn = None ## ""

    def forward(self, x, cur_data_fidelity, cur_prior, cur_params, y, physics): 
        ## Proximal data fidelity step
        z = cur_data_fidelity.prox(
            x["est"][0],
            y,
            physics,
            gamma=cur_params["stepsize"]
        )
        ## Proximal denoiser step
        x = cur_prior.prox(
            z,
            physics.diff_physics.ts
        )
        
        return {"est":(x,z)}