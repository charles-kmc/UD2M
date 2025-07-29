import torch


def default_preprocessing(y, physics):
    r"""
    Default preprocessing function for spectral methods.

    The output of the preprocessing function is given by:

    .. math::
        \max(1 - 1/y, -5).

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Instance of the physics modeling the forward matrix.

    :return: The preprocessing function values evaluated at y.
    """
    return torch.max(1 - 1 / y, torch.tensor(-5.0))

def spectral_methods(
    y: torch.Tensor,
    physics,
    x=None,
    n_iter=50,
    preprocessing=default_preprocessing,
    lamb=10.0,
    x_true=None,
    early_stop: bool = True,
    rtol: float = 1e-5,
    verbose: bool = False,
    *wargs,
    **kwargs,
):
    r"""
    Utility function for spectral methods.

    This function runs the Spectral Methods algorithm to find the principal eigenvector of the regularized weighted covariance matrix:
    
    .. math::
        \begin{equation*}
        M = \conj{B} \text{diag}(T(y)) B + \lambda I,
        \end{equation*}
    
    where :math:`B` is the linear operator of the phase retrieval class, :math:`T(\cdot)` is a preprocessing function for the measurements, and :math:`I` is the identity matrix of corresponding dimensions. Parameter :math:`\lambda` tunes the strength of regularization.

    To find the principal eigenvector, the function runs power iteration which is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        x_{k+1} &= M x_k \\
        x_{k+1} &= \frac{x_{k+1}}{\|x_{k+1}\|},
        \end{aligned}
        \end{equation*}
    """
    if x is None:
        # always use randn for initial guess, never use rand!
        x = torch.randn(
            physics.input_shape,
            dtype=physics.dtype,
            device=physics.device,
        )
    norm_x = torch.sqrt(y.sum())
    x = x.to(torch.cfloat)
    y = y / torch.mean(y)
    diag_T = preprocessing(y, physics)
    diag_T = diag_T.to(torch.float)
    for i in range(n_iter):
        x_new = physics.A(x)
        # x_new = diag_T * x_new
        x_new = physics.AT(x_new)
        x_new = x_new + lamb * x
        x_new = x_new / torch.linalg.norm(x_new)
        
        if early_stop:
            if torch.linalg.norm(x_new - x) / torch.linalg.norm(x) < rtol:
                if verbose:
                    print(f"Power iteration early stopped at iteration {i}.")
                break
        x = x_new
    #! change the norm of x so that it matches the norm of true x
    x = x * norm_x
    
    return x
