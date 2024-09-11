import torch 
from deepinv.physics import DecomposablePhysics
## Currently assumes physics is of the type div.Decomposable Physics, which
## allows efficient computation of the prox of the data fidelity term 

# Function to initialize hqs given the observation 
# obs = (y, x_t)
def init_hqs_diff(obs, physics:DecomposablePhysics):  # Initialise HQS using maximum likelihood
    y, xt = obs 
    sy = physics.y_physics.noise_model.sigma
    st = physics.diff_physics.sigma
    ##  Compute (Lipschitz regularized) maximul likelihood estimate of x
    ## Move to decomposition space
    xt = xt / physics.diff_physics.sqrt_alphas_cumprod[physics.diff_physics.ts]
    b = physics.y_physics.A_adjoint(y)/sy**2 + xt / st**2 
    if isinstance(physics.y_physics.mask, float):
        scaling = physics.y_physics.mask**2/ physics.y_physics.noise_model.sigma**2 + 1 / physics.diff_physics.sigma**2 * xt
    else:  # Dinv uses rfft for singular value decomposition of circular blur
        scaling = torch.conj(physics.y_physics.mask) * physics.y_physics.mask/ physics.y_physics.noise_model.sigma**2 
        scaling = scaling.expand(physics.diff_physics.sigma.shape[0],*scaling.shape[1:])
        scaling = scaling + 1 / physics.diff_physics.sigma.clone().unsqueeze(-1)**2 
    ## Remove large eigenvalues from pseudoinverse
    mask = torch.zeros_like(scaling)
    mask[scaling > 1e-1] = 1/scaling[scaling>1e-1]
    ## Estimate pseudoinverse
    init = physics.y_physics.V(physics.y_physics.V_adjoint(b) * mask).clip(0,1)
    return {"est":(init, init)}

# def init_hqs_diff(obs, physics:DecomposablePhysics): # Initialise HQS from y
#     y, xt = obs 
#     return {"est":(y, y)}

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
        )  # Computes prox operator of \norm{Ax - y}^2/2\sigma_y^2 + \norm{x_t/\sqrt{\alpha_t} - x_0}^2/2\sigma_t^2
        ## Proximal denoiser step
        x = cur_prior.prox(
            z,
            physics.diff_physics.ts,
            eta=cur_params["eta"]
        )
        x = x.clip(0,1)
        
        return {"est":(x,z)}  