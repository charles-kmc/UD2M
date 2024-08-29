import deepinv 
import torch

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

class SampleIter(deepinv.optim.FixedPoint):
    """
    Sampling iterations module.

    This module implements an iterative sampling algorithm given a specific iterator (e.g.
    ULA, SKROCK iteration.
    """

    def __init__(
        self,
        iterator=None,
        update_params_fn=None,
        update_data_fidelity_fn=None,
        update_prior_fn=None,
        init_iterate_fn=None,
        weighted_sum=None,
        init_metrics_fn=None,
        update_metrics_fn=None,
        check_iteration_fn=None,
        max_iter=50
    ):
        super().__init__(
            iterator=iterator,
            update_params_fn=update_params_fn,
            update_data_fidelity_fn=update_data_fidelity_fn,
            update_prior_fn=update_prior_fn,
            init_iterate_fn=init_iterate_fn,
            init_metrics_fn=init_metrics_fn,
            update_metrics_fn=update_metrics_fn,
            check_iteration_fn=check_iteration_fn,
            max_iter=max_iter,
            early_stop=False
        )
        self.weighted_sum=weighted_sum
    
    def single_iteration(self, X, it, *args, **kwargs):

        cur_params = self.update_params_fn(it) if self.update_params_fn else None
        cur_data_fidelity = (
            self.update_data_fidelity_fn(it) if self.update_data_fidelity_fn else None
        )
        cur_prior = self.update_prior_fn(it) if self.update_prior_fn else None
        X_prev = X
        X = self.iterator(X_prev, cur_data_fidelity, cur_prior, cur_params, *args)
        if self.weighted_sum:
            X["weighted_sum"] = X_prev["weighted_sum"] + cur_params["weight"] * X["est"][0]
            X["tot_weight"] = X_prev["tot_weight"] + cur_params["weight"]
        return X 

class BaseSample(deepinv.optim.BaseOptim):
    def __init__(
            self,
            iterator,
            weighted_sum=False,
            **kwargs,
    ):
        super(BaseSample,self).__init__(iterator,**kwargs)
        self.weighted_sum = weighted_sum
        if self.weighted_sum:
            self.get_output = lambda x: x["weighted_sum"]/x["tot_weight"]
        else:
            self.get_output = lambda agg: agg["est"][0]
        self.fixed_point = SampleIter(
            iterator=iterator,
            update_params_fn=self.update_params_fn,
            update_data_fidelity_fn=self.update_data_fidelity_fn,
            update_prior_fn=self.update_prior_fn,
            weighted_sum=weighted_sum,
            init_iterate_fn=self.init_iterate_fn,
            init_metrics_fn=self.init_metrics_fn,
            update_metrics_fn=self.update_metrics_fn,
            check_iteration_fn=self.check_iteration_fn,
            max_iter=self.max_iter
        )

class BaseUnfoldLP(BaseSample):
    r"""
    Base class for unfolded algorithms. Child of :class:`deepinv.optim.BaseOptim`.

    Enables to turn any iterative optimization algorithm into an unfolded algorithm, i.e. an algorithm
    that can be trained end-to-end, with learnable parameters. Recall that the algorithms have the
    following form (see :meth:`deepinv.optim.OptimIterator`):

    .. math::
        \begin{aligned}
        z_{k+1} &= \operatorname{step}_f(x_k, z_k, y, A, \gamma, ...)\\
        x_{k+1} &= \operatorname{step}_g(x_k, z_k, y, A, \lambda, \sigma, ...)
        \end{aligned}

    where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are learnable modules.
    These modules encompass trainable parameters of the algorithm (e.g. stepsize :math:`\gamma`, regularization parameter :math:`\lambda`, prior parameter (`g_param`) :math:`\sigma` ...)
    as well as trainable priors (e.g. a deep denoiser).

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
        dictionary for the :meth:`deepinv.optim.OptimIterator` class.
        This does not encompass the trainable weights of the prior module.
    :param torch.device device: Device on which to perform the computations. Default: ``torch.device("cpu")``.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param kwargs: Keyword arguments to be passed to the :class:`deepinv.optim.BaseOptim` class.
    """

    def __init__(
        self,
        iterator,
        params_algo={"lambda": 1.0, "stepsize": 1.0},
        data_fidelity=None,
        prior=None,
        max_iter=5,
        trainable_params=["lambda", "stepsize"],
        log_trainable_params = ["lambda", "stepsize"],
        device=torch.device("cpu"),
        denoise_final = False,
        tied_params = [],
        *args,
        **kwargs,
    ):
        super().__init__(
            iterator,
            max_iter=max_iter,
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            **kwargs,
        )
        self.denoise_final = denoise_final and ('denoiser' in dir(prior))
        if self.denoise_final:
            self.sigma_pp = torch.nn.Parameter(torch.tensor(1e-2).log())
            self.get_output = lambda x: prior.denoiser(x["est"][0], self.sigma_pp.exp().clip(0,0.5))
        # Each parameter in `init_params_algo` is a list, which is converted to a `nn.ParameterList` if they should be trained.
        for param_key in trainable_params:
            if param_key in self.init_params_algo.keys():
                param_value = self.init_params_algo[param_key]
                if param_key in log_trainable_params:
                    self.init_params_algo[param_key] = nn.ParameterList(
                        [nn.Parameter(torch.tensor(el).log().to(device))  for el in param_value]
                    )
                else:
                    self.init_params_algo[param_key] = nn.ParameterList(
                        [nn.Parameter(torch.tensor(el).to(device))  for el in param_value]
                    )
        self.log_trainable_params = log_trainable_params
        self.init_params_algo = nn.ParameterDict(self.init_params_algo)
        self.params_algo = self.init_params_algo.copy()
        # The prior (list of instances of :class:`deepinv.optim.Prior`) is converted to a `nn.ModuleList` to be trainable.
        self.prior = nn.ModuleList(self.prior)
        self.data_fidelity = nn.ModuleList(self.data_fidelity)
        self.tied_params = tied_params

        # Control to make sure all parameters supplied as tied weights are recoded in self.tied_params
        if self.max_iter > 1:
            for k, v in self.init_params_algo.items():
                if k not in self.tied_params and len(v) == 1:
                    self.tied_params.append(k)
    
    def update_params_fn(self, it):
        cur_params_dict =  super().update_params_fn(it)
        for param_key in self.log_trainable_params:
            if param_key in cur_params_dict.keys():
                cur_params_dict[param_key] = cur_params_dict[param_key].exp()
        return cur_params_dict
    
    def add_new_level(self):
        params = []
        for k, v in self.params_algo.items():
            if k in self.tied_params:
                break
            else:
                if hasattr(v[-1], 'requires_grad') and v[-1].requires_grad:
                    self.init_params_algo[k].append(v[-1].detach().clone().requires_grad_())
                    params.append(self.init_params_algo[k][-1])
                else: 
                    self.init_params_algo[k].append(v[-1])
        self.max_iter += 1
        self.fixed_point.max_iter = self.max_iter
        return params

    def log_param_dict(self, physics):
        log = {}
        for it in range(self.max_iter):
            for k,v in self.update_params_fn(it).items():
                if isinstance(v, float) or isinstance(v, int) or v is None:  # Removes instances which are not trainable
                    continue
                if v.numel() <= 1:
                    log[k + '/' + str(it)] = v
                else: 
                    try:
                        v_im = v.squeeze() if len(v.shape)>3 else v 
                        v_im = v_im.unsqueeze(0) if len(v.shape) < 3 else v_im

                        v_m = v_im.min()
                        v_M = v_im.max()
                        v_l = (v_im - v_m + 1e-8).log()
                        v_lm = v_l.min()
                        v_lM = v_l.max()
                        v_im = (v_l - v_lm)/(v_lM - v_lm)
                        log[k + '/' + str(it)] = v.mean()
                        log[k + '_im' + '/' + str(it) ] = wandb.Image(v_im, caption='min = {}, max = {}'.format(v_m,v_M))
                    except:
                        pass
        if self.denoise_final:
            log['sigma_postprocessing'] = self.sigma_pp.exp().item()
        return log

class EndToEndSampler(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network 


    def forward(self, y, physics):
        return self.network(y,sigma=None)