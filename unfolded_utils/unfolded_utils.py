import torch
from deepinv.optim.optimizers import create_iterator
from pathlib import Path 
import os 
import wandb
import numpy as np
import torchvision
from tqdm import tqdm
from .train_weights import train_W
from unfolded_models.unrolled_sampling import BaseUnfoldLP

def init_params_algo(iterator, prior, n_params, img_size, matrix_rho=False, physics=None, train_w_args ={}, weights = "final", learn_weights = False, tied_weight = True):
    if not isinstance(iterator, str):
        try:
            iterator = str(iterator.__name__)[:-8]
        except:
            raise ValueError("Iterator {} is not valid".format(iterator))
    if hasattr(physics, 'noise_model'):
        try:
            sigma = physics.noise_model.sigma.data 
        except:
            raise ValueError("Physics does not have attribute sigma, using pre-determined value")
    else: 
        sigma = 0.05
    if len(img_size) == 3:
        img_size = (1, *img_size)
    stepsize = [0.5*sigma**2]*n_params  # Sequence of step_sizes 
    sigma_denoiser = [1e-2]*n_params  # Sequence of noise_levels
    lda = [1.]*n_params
    beta = [1.]*n_params
    tied_params = []
    rho = torch.tensor(sigma**2) if not hasattr(prior, "gamma_x") else prior.gamma_x.data
    if matrix_rho:
        rho = rho * torch.ones(img_size)
    params_algo = {}
    params_algo["stepsize"] = stepsize
    if iterator in ["PGD", "ADMM", "DRS", "HQS", "Latent_ULA", "Latent_GD", "ULA", "LatentProx", "LatentProx_GD","Latent_Gibbs", "Latent_Gibbs_GD","EmbeddedULA", "EmbeddedGD"]:
        params_algo["g_param"] = sigma_denoiser
    if iterator in ["PGD", "ADMM", "DRS", "HQS", "ULA","Latent_ULA", "Latent_GD"]:
        params_algo["lambda"] = lda
    if iterator in ["ADMM", "DRS"]:
        params_algo["beta"] = beta
    if iterator in ["Latent_ULA", "Latent_GD", "LatentProx", "LatentProx_GD","Latent_Gibbs", "Latent_Gibbs_GD"]:
        params_algo["rho"] = [rho]*n_params
    if "Precond" in iterator:
        params_algo["stepsize"] = [0.5 if "Score" not in iterator else 0.005]*n_params
        params_algo["theta"] = [1e-2]*(1 if tied_weight else n_params)
        params_algo["lambda"] = lda 
        params_algo["g_param"] = sigma_denoiser
    if "LISTA" in iterator:
        params_algo = {"g_param":sigma_denoiser,"lambda":lda}
        if "LISTA_CP" in iterator:
            W_cp = train_W(**train_w_args)
            params_algo["stepsize"] = [1.]*n_params
            params_algo["weight_CP"] = [W_cp]
            if tied_weight:
                tied_params.append("weight_CP")
            else: 
                params_algo["weight_CP"] = [W_cp]*n_params

    trainable_params = list(params_algo.keys())
    # if "pnp" not in prior and "lambda" in trainable_params: 
    #     params_algo["lambda"] = [1.]*n_params
    #     trainable_params.remove("lambda")
    if "LISTA_CP_noweight" in iterator:
        trainable_params.remove("weight_CP")
    params_algo["weight"] = [0]*(n_params-1) + [1] if weights=="final" else [1]*n_params
    if learn_weights:
        trainable_params.append["weight"]
    return params_algo, trainable_params, tied_params

def custom_init_y(y, physics):
    return {"est":(y,y)}

def custom_init_y_adj(y, physics):
    Aty = physics.A_adjoint(y)
    return {"est":(Aty,Aty)}

def unfolded_builderLP(
    iteration,
    init_adj = False,
    weighted_sum=False,
    params_algo={"lambda": 1.0, "stepsize": 1.0},
    data_fidelity=None,
    prior=None,
    max_iter=5,
    trainable_params=["lambda", "stepsize"],
    log_trainable_params = ["lambda", "stepsize"],
    device=torch.device("cpu"),
    F_fn=None,
    g_first=False,
    verbose = False,
    denoise_final=False,
    custom_init = None,
    **kwargs,
):
    r"""
    Helper function for building an unfolded architecture.

    :param str, deepinv.optim.OptimIterator iteration: either the name of the algorithm to be used,
        or directly an optim iterator.
        If an algorithm name (string), should be either ``"GD"`` (gradient descent), ``"PGD"`` (proximal gradient descent),
        ``"ADMM"`` (ADMM),
        ``"HQS"`` (half-quadratic splitting), ``"CP"`` (Chambolle-Pock) or ``"DRS"`` (Douglas Rachford). See
        <optim> for more details.
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
        e.g. the stepsize, regularisation parameter, denoising standard deviation.
        Each value of the dictionary can be either Iterable (distinct value for each iteration) or
        a single float (same value for each iteration).
        Default: ``{"stepsize": 1.0, "lambda": 1.0}``. See :any:`optim-params` for more details.
    :param list, deepinv.optim.DataFidelity: data-fidelity term.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :meth:`deepinv.optim.DataFidelity` (distinct data-fidelity for each iteration). Default: ``None``.
    :param list, deepinv.optim.Prior prior: regularization prior.
        Either a single instance (same prior for each iteration) or a list of instances of
        deepinv.optim.Prior (distinct prior for each iteration). Default: ``None``.
    :param int max_iter: number of iterations of the unfolded algorithm. Default: 5.
    :param list trainable_params: List of parameters to be trained. Each parameter should be a key of the ``params_algo``
        dictionary for the :class:`deepinv.optim.OptimIterator` class.
        This does not encompass the trainable weights of the prior module.
    :param callable F_fn: Custom user input cost function. default: None.
    :param torch.device device: Device on which to perform the computations. Default: ``torch.device("cpu")``.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param kwargs: additional arguments to be passed to the :meth:`BaseOptim` class.
    :return: an unfolded architecture (instance of :meth:`BaseUnfold`).

    |sep|

    :Example:

    .. doctest::

        >>> import torch
        >>> import deepinv as dinv
        >>>
        >>> # Create a trainable unfolded architecture
        >>> model = dinv.unfolded.unfolded_builder(
        ...     iteration="PGD",
        ...     data_fidelity=dinv.optim.L2(),
        ...     prior=dinv.optim.PnP(dinv.models.DnCNN(in_channels=1, out_channels=1, train=True)),
        ...     params_algo={"stepsize": 1.0, "g_param": 1.0},
        ...     trainable_params=["stepsize", "g_param"]
        ... )
        >>> # Forward pass
        >>> x = torch.randn(1, 1, 16, 16)
        >>> physics = dinv.physics.Denoising()
        >>> y = physics(x)
        >>> x_hat = model(y, physics)


    """
    kwargs["custom_init"] = custom_init_y_adj if init_adj else custom_init_y
    kwargs["custom_init"] = custom_init if custom_init else kwargs["custom_init"]
    iterator = create_iterator(iteration, prior=prior, F_fn=F_fn, g_first=g_first)
    return BaseUnfoldLP(
        iterator,
        max_iter=max_iter,
        trainable_params=trainable_params,
        log_trainable_params=log_trainable_params,
        has_cost=iterator.has_cost,
        data_fidelity=data_fidelity,
        weighted_sum=weighted_sum,
        prior=prior,
        params_algo=params_algo,
        device=device,
        denoise_final=denoise_final,
        verbose=verbose,
        **kwargs,
    )




    
    

