# from sklearn.preprocessing import OrdinalEncoder
import torch as th
import numpy as np
import logging

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde

def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)
    

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1


    # def sample(self, x1):
    #     """Sampling x0 & t based on shape of x1 (if needed)
    #       Args:
    #         x1 - data point; [batch, *dim]
    #     """
        
    #     x0 = th.randn_like(x1)
    #     t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
    #     t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
    #     t = t.to(x1)
    #     return t, x0, x1
    

    # def training_losses(
    #     self, 
    #     model,  
    #     x1, 
    #     model_kwargs=None
    # ):
    #     """Loss for training the score model
    #     Args:
    #     - model: backbone model; could be score, noise, or velocity
    #     - x1: datapoint
    #     - model_kwargs: additional arguments for the model
    #     """
    #     if model_kwargs == None:
    #         model_kwargs = {}
        
    #     t, x0, x1 = self.sample(x1)
    #     t, xt, ut = self.path_sampler.plan(t, x0, x1)
    #     model_output = model(xt, t, **model_kwargs)
    #     B, *_, C = xt.shape
    #     assert model_output.size() == (B, *xt.size()[1:-1], C)

    #     terms = {}
    #     terms['pred'] = model_output
    #     if self.model_type == ModelType.VELOCITY:
    #         terms['loss'] = mean_flat(((model_output - ut) ** 2))
    #     else: 
    #         _, drift_var = self.path_sampler.compute_drift(xt, t)
    #         sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
    #         if self.loss_type in [WeightType.VELOCITY]:
    #             weight = (drift_var / sigma_t) ** 2
    #         elif self.loss_type in [WeightType.LIKELIHOOD]:
    #             weight = drift_var / (sigma_t ** 2)
    #         elif self.loss_type in [WeightType.NONE]:
    #             weight = 1
    #         else:
    #             raise NotImplementedError()
            
    #         if self.model_type == ModelType.NOISE:
    #             terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
    #         else:
    #             terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
                
    #     return terms
    
    def sample(self, x1, num_steps, loss_type):
        """Sampling x0 & t based on shape of x1 (if needed)
            Args:
            x1 - data point; [batch, *dim]
        """
        
        x0 = th.randn_like(x1)

        if loss_type == 'sdei':
            tau = None
            d = 1 / num_steps * th.rand((x1.shape[0],), device=x1.device)
            t = th.rand((x1.shape[0],), device=x1.device) * (1 - d)
            t_floor = th.floor(t * num_steps) / num_steps
            s = d + t_floor
            t = t_floor
        elif loss_type == 'stei':
            tau = th.rand((x1.shape[0],), device=x1.device)
            d = 1 / num_steps * th.rand((x1.shape[0],), device=x1.device)
            t = th.rand((x1.shape[0],), device=x1.device) * (1 - d)
            s = d + t
        elif loss_type == 'sdee':
            tau = None
            d = 1 / num_steps * th.rand((x1.shape[0],))
            t_ = th.rand((x1.shape[0],)) * (1 - d)
            s_ = d + t_
            mask = th.randint(0, 2, (x1.shape[0],))
            s = s_ * mask + t_ * (1 - mask)
            t = t_ * mask + s_ * (1 - mask)
        elif loss_type == 'stee':
            tau = None
            d = 1 / num_steps * th.rand((x1.shape[0],))
            t_ = th.rand((x1.shape[0],)) * (1 - d)
            s_ = d + t_
            mask = th.randint(0, 2, (x1.shape[0],))
            s = s_ * mask + t_ * (1 - mask)
            t = t_ * mask + s_ * (1 - mask)
        else:
            raise NotImplementedError
        
        r = th.rand((x1.shape[0],), device=x1.device)
        r = t + r * (s - t)

        return tau, r, s, t, x0, x1

    def training_losses(
        self, 
        model,
        teacher,
        x1,
        online_cfg=None,
        num_steps=4,
        loss_type='sdei',
        model_kwargs=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        dropout_prob = 0.1
        num_classes = 1000

        if model_kwargs == None:
            model_kwargs = {}
        
        tau, r, s, t, x0, x1 = self.sample(x1, num_steps, loss_type)

        y = model_kwargs['y']
        drop_ids = th.rand(y.shape[0], device=y.device) < dropout_prob
        dropped_y = th.where(drop_ids, num_classes, y)

        t_ = expand_t_like_x(t, x1)
        r_ = expand_t_like_x(r, x1)
        s_ = expand_t_like_x(s, x1)

        if loss_type == 'sdei':
            xt = t_ * x1 + (1 - t_) * x0
            if online_cfg is None:
                with th.no_grad():
                    xr_hat = xt + (r_ - t_) * model(xt, t, r, dropped_y, cfg=None)
                    vr_hat = teacher(xr_hat, r, r, dropped_y, cfg=None)
                f_ts = model(xt, t, s, dropped_y, cfg=None)
            else:
                online_cfg = online_cfg.split('-')
                if len(online_cfg) == 1:
                    cfg = float(online_cfg[0]) * x1.new_ones(x1.shape[0])
                    assert th.all(cfg > 1)
                elif len(online_cfg) == 2:
                    cfg_min = float(online_cfg[0])
                    cfg_max = float(online_cfg[1])
                    assert cfg_min < cfg_max
                    cfg = th.rand((x1.shape[0],)) * (cfg_max - cfg_min) + cfg_min
                    cfg = cfg.to(x1)
                    assert th.all(cfg >= 1)
                else:
                    raise NotImplementedError
                
                with th.no_grad():
                    xr_hat = xt + (r_ - t_) * model(xt, t, r, y, cfg=cfg)
                    xr_hat = th.cat([xr_hat, xr_hat], dim=0)
                    y_null = th.tensor([1000] * x1.shape[0], device=x1.device)
                    y_ = th.cat([y, y_null], 0)
                    r_ = th.cat([r, r], 0)
                    vr_hat = teacher.forward_with_cfg(xr_hat, r_, r_, y=y_, cfg_scale=cfg).chunk(2, dim=0)[0]
            
                f_ts = model(xt, t, s, y, cfg=cfg)

            terms = {}
            terms['loss'] = mean_flat(((f_ts - vr_hat) ** 2))

        elif loss_type == 'stei':
            xt = t_ * x1 + (1 - t_) * x0
            tau_ = expand_t_like_x(tau, x1)
            xtau = tau_ * x1 + (1 - tau_) * x0
            utau = x1 - x0
            vtau = model(xtau, tau, tau, dropped_y, cfg=x1.new_ones(x1.shape[0]))
            if online_cfg is None:
                with th.no_grad():
                    xr_hat = xt + (r_ - t_) * model(xt, t, r, dropped_y, cfg=None)
                    vr_hat = teacher(xr_hat, r, r, dropped_y, cfg=None)
                f_ts = model(xt, t, s, dropped_y, cfg=None)
            else:
                online_cfg = online_cfg.split('-')
                if len(online_cfg) == 1:
                    cfg = float(online_cfg[0]) * x1.new_ones(x1.shape[0])
                    assert th.all(cfg > 1)
                elif len(online_cfg) == 2:
                    cfg_min = float(online_cfg[0])
                    cfg_max = float(online_cfg[1])
                    assert cfg_min < cfg_max
                    cfg = th.rand((x1.shape[0],)) * (cfg_max - cfg_min) + cfg_min
                    cfg = cfg.to(x1)
                    assert th.all(cfg >= 1)
                else:
                    raise NotImplementedError
                
                with th.no_grad():
                    xr_hat = xt + (r_ - t_) * model(xt, t, r, y, cfg=cfg)

                    xr_hat = th.cat([xr_hat, xr_hat], dim=0)
                    y_null = th.tensor([1000] * x1.shape[0], device=x1.device)
                    y_ = th.cat([y, y_null], 0)
                    r_ = th.cat([r, r], 0)
                    vr_hat = teacher.forward_with_cfg(xr_hat, r_, r_, y=y_, cfg_scale=cfg).chunk(2, dim=0)[0]
            
                f_ts = model(xt, t, s, y, cfg=cfg)

            terms = {}
            terms['loss'] = mean_flat(((f_ts - vr_hat) ** 2)) + mean_flat(((vtau - utau) ** 2))
        
        elif loss_type == 'sdee':
            xr = r_ * x1 + (1 - r_) * x0
            if online_cfg is None:
                with th.no_grad():
                    xt_hat = xr + (t_ - r_) * model(xr, r, t, dropped_y, cfg=None)
                    vr_hat = teacher(xr_hat, r, r, dropped_y, cfg=None)
                f_ts = model(xt_hat, t, s, dropped_y, cfg=None)
            else:
                online_cfg = online_cfg.split('-')
                if len(online_cfg) == 1:
                    cfg = float(online_cfg[0]) * x1.new_ones(x1.shape[0])
                    assert th.all(cfg > 1)
                elif len(online_cfg) == 2:
                    cfg_min = float(online_cfg[0])
                    cfg_max = float(online_cfg[1])
                    assert cfg_min < cfg_max
                    cfg = th.rand((x1.shape[0],)) * (cfg_max - cfg_min) + cfg_min
                    cfg = cfg.to(x1)
                    assert th.all(cfg >= 1)
                else:
                    raise NotImplementedError
                
                with th.no_grad():
                    xt_hat = xr + (t_ - r_) * model(xr, r, t, y, cfg=cfg)
                    xr_cat = th.cat([xr, xr], dim=0)
                    y_null = th.tensor([1000] * x1.shape[0], device=x1.device)
                    y_cat = th.cat([y, y_null], 0)
                    r_cat = th.cat([r, r], 0)
                    vr_hat = teacher.forward_with_cfg(xr_cat, r_cat, r_cat, y=y_cat, cfg_scale=cfg).chunk(2, dim=0)[0]
                
                f_ts = model(xt_hat, t, s, y, cfg=cfg)

            terms = {}
            terms['loss'] = mean_flat(((f_ts - vr_hat) ** 2))
        
        elif loss_type == 'stee':
            xr = r_ * x1 + (1 - r_) * x0
            ur = x1 - x0
            xt_hat = xr + (t_ - r_) * model(xr, r, t, dropped_y, cfg=None).detach()
            f_ts = model(xt_hat, t, s, dropped_y, cfg=None)
            
            terms = {}
            terms['loss'] = mean_flat(((f_ts - ur) ** 2))
        
        else:
            raise NotImplementedError
        
        return terms
    

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn
    

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        
        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion
    
    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        
        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)
        
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn