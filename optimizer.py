# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

from torch.optim.optimizer import Optimizer
import math
import torch

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Any, Callable, Optional
import torch.optim

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class MADGRAD(torch.optim.Optimizer):
    """
    MADGRAD_: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
    Optimization.
    .. _MADGRAD: https://arxiv.org/abs/2101.11075
    MADGRAD is a general purpose optimizer that can be used in place of SGD or
    Adam may converge faster and generalize better. Currently GPU-only.
    Typically, the same learning rate schedule that is used for SGD or Adam may
    be used. The overall learning rate is not comparable to either method and
    should be determined by a hyper-parameter sweep.
    MADGRAD requires less weight decay than other methods, often as little as
    zero. Momentum values used for SGD or Adam's beta1 should work here also.
    On sparse problems both weight_decay and momentum should be set to 0.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate (default: 1e-2).
        momentum (float):
            Momentum value in  the range [0,1) (default: 0.9).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-6).
    """

    def __init__(
            self, params: _params_t, lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0, eps: float = 1e-6,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1]")
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if eps < 0:
            raise ValueError(f"Eps must be non-negative")

        defaults = dict(lr=lr, eps=eps, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return False

    @property
    def supports_flat_params(self) -> bool:
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # step counter must be stored in state to ensure correct behavior under
        # optimizer sharding
        if 'k' not in self.state:
            self.state['k'] = torch.tensor([0], dtype=torch.long)
        k = self.state['k'].item()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"] + eps
            decay = group["weight_decay"]
            momentum = group["momentum"]

            ck = 1 - momentum
            lamb = lr * math.pow(k + 1, 0.5)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if "grad_sum_sq" not in state:
                    state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                    state["s"] = torch.zeros_like(p.data).detach()
                    if momentum != 0:
                        state["x0"] = torch.clone(p.data).detach()

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError("momentum != 0 is not compatible with sparse gradients")

                grad_sum_sq = state["grad_sum_sq"]
                s = state["s"]

                # Apply weight decay
                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")

                    grad.add_(p.data, alpha=decay)

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_val = grad._values()

                    p_masked = p.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    # Compute x_0 from other known quantities
                    rms_masked_vals = grad_sum_sq_masked._values().pow(1 / 3).add_(eps)
                    x0_masked_vals = p_masked._values().addcdiv(s_masked._values(), rms_masked_vals, value=1)

                    # Dense + sparse op
                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=lamb)
                    grad_sum_sq_masked.add_(grad_sq, alpha=lamb)

                    rms_masked_vals = grad_sum_sq_masked._values().pow_(1 / 3).add_(eps)

                    s.add_(grad, alpha=lamb)
                    s_masked._values().add_(grad_val, alpha=lamb)

                    # update masked copy of p
                    p_kp1_masked_vals = x0_masked_vals.addcdiv(s_masked._values(), rms_masked_vals, value=-1)
                    # Copy updated masked p to dense p using an add operation
                    p_masked._values().add_(p_kp1_masked_vals, alpha=-1)
                    p.data.add_(p_masked, alpha=-1)
                else:
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3).add_(eps)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    # Accumulate second moments
                    grad_sum_sq.addcmul_(grad, grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3).add_(eps)

                    # Update s
                    s.data.add_(grad, alpha=lamb)

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)

        self.state['k'] += 1
        return loss


class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3,  # lr
                 alpha=0.5, k=6, n_sma_threshhold=5,  # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 use_gc=True, gc_conv_only=False
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 ):

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=n_sma_threshhold,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = n_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.use_gc = use_gc

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if self.use_gc and self.gc_gradient_threshold == 1:
            print(f"GC applied to both conv and fc layers")
        elif self.use_gc and self.gc_gradient_threshold == 3:
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a
        # callable closure. Uncomment if you need to use the actual closure...

        # if closure is not None:
        # loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    # print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                state['step'] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = n_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = n_sma
                    if n_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (n_sma - 4) / (n_sma_max - 4) * (n_sma - 2) / n_sma * n_sma_max / (
                                    n_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # apply lr
                if n_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']  # get access to slow param tensor
                    slow_p.add_(self.alpha, p.data - slow_p)  # (fast weights - slow weights) * alpha
                    p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor

        return loss
    

# Ranger21 - @lessw2020  and @NestorDemeure
# with contributions from:
# @BrianPugh
# @Kayuksel
# @TheZothen

# core components based on:

# MADGRAD: https://arxiv.org/abs/2101.11075

# warmup:  https://arxiv.org/abs/1910.04209v3

# stable weight decay: https://arxiv.org/abs/2011.11152v3

# Gradient Centralization: https://arxiv.org/abs/2004.01461v2

# positive negative momentum:  https://arxiv.org/abs/2103.17182

# adaptive gradient clipping: https://arxiv.org/abs/2102.06171)
#   big thanks to @kayuksel for suggestion to include agc, and initial code,
#   @lucidrains and @rwightman for additional code impl reference

# softplus transformation to denom:  https://arxiv.org/abs/1908.00700

# lookahead:
# Lookahead Optimizer: https://arxiv.org/abs/1907.08610

# norm loss: https://arxiv.org/abs/2103.06583v1
# big thanks to Theodoros Georgiou for TF code implementation, and he and team for inventing norm loss

# flat lr + cosine decay: original work 2019 (fastai team)

# Chebyshev fractal steps:

# This space for rent - send in your improvements!


import torch
import torch.optim as TO
import torch.nn.functional as F

import math
import collections

# this is to support showing the lr curves after a training run.
import matplotlib.pyplot as plt

import copy
from torch import linalg as LA

import numpy as np


def cheb_steps(m, M, T):
    C, R = (M + m) / 2.0, (M - m) / 2.0
    thetas = (np.arange(T) + 0.5) / T * np.pi
    return 1.0 / (C - R * np.cos(thetas))


def cheb_perm(T):
    perm = np.array([0])
    while len(perm) < T:
        perm = np.vstack([perm, 2 * len(perm) - 1 - perm]).T.flatten()
    return perm


def get_chebs(num_epochs):
    num_epochs = num_epochs - 2
    steps = cheb_steps(0.1, 1, num_epochs)
    perm = cheb_perm(num_epochs)
    cheb_schedule = steps[perm]
    print(f"cheb schedule made with len {len(cheb_schedule)}")
    return cheb_schedule


def normalize_gradient(x, use_channels=False, epsilon=1e-8):
    """  use stdev to normalize gradients """
    size = x.dim()
    # print(f"size = {size}")

    if (size > 1) and use_channels:
        s = x.std(dim=tuple(range(1, size)), keepdim=True) + epsilon
        # print(f"s = {s}")
        x.div_(s)  # , keepdim=True)

    elif torch.numel(x) > 2:
        s = x.std() + epsilon
        x.div_(s)  # , keepdim=True)
    return x


def centralize_gradient(x, gc_conv_only=False):
    """credit - https://github.com/Yonghongwei/Gradient-Centralization """

    size = x.dim()
    # print(f"size = {size}")

    if gc_conv_only:
        if size > 3:
            x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
    else:
        if size > 1:
            x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
    return x


class Ranger21(TO.Optimizer):
    def __init__(
        self,
        params,
        lr,
        lookahead_active=True,
        lookahead_mergetime=5,
        lookahead_blending_alpha=0.5,
        lookahead_load_at_validation=False,
        use_madgrad=False,
        use_adabelief=False,
        softplus=True,
        beta_softplus=50,
        using_gc=True,
        using_normgc=True,
        gc_conv_only=False,
        normloss_active=True,
        normloss_factor=1e-4,
        use_adaptive_gradient_clipping=True,
        agc_clipping_value=1e-2,
        agc_eps=1e-3,
        betas=(0.9, 0.999),  # temp for checking tuned warmups
        momentum_type="pnm",
        pnm_momentum_factor=1.0,
        momentum=0.9,
        eps=1e-8,
        num_batches_per_epoch=1180,
        num_epochs=100,
        use_cheb=False,
        use_warmup=False,
        num_warmup_iterations=None,
        warmdown_active=False,
        warmdown_start_pct=0.72,
        warmdown_min_lr=3e-5,
        weight_decay=1e-4,
        decay_type="stable",
        warmup_type="linear",
        warmup_pct_default=0.22,
        logging_active=True,
    ):

        # todo - checks on incoming params
        defaults = dict(
            lr=lr, momentum=momentum, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

        # core
        self.logging = logging_active

        # engine
        self.use_madgrad = use_madgrad

        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_epochs = num_epochs

        if not self.use_madgrad:
            self.core_engine = "AdamW"
        else:
            self.core_engine = "madgrad"

        # ada belief:
        self.use_adabelief = use_adabelief

        # eps
        self.eps = eps

        # softplus for denom
        self.softplus = softplus
        self.beta_softplus = beta_softplus

        # norm loss
        self.normloss_active = normloss_active
        self.normloss_factor = normloss_factor

        # lookahead
        self.lookahead_active = lookahead_active
        self.lookahead_mergetime = lookahead_mergetime
        self.lookahead_step = 0
        self.lookahead_alpha = lookahead_blending_alpha

        self.lookahead_validation_load = lookahead_load_at_validation

        # agc
        self.agc_active = use_adaptive_gradient_clipping
        self.agc_clip_val = agc_clipping_value
        self.agc_eps = agc_eps

        # chebs
        self.use_cheb = use_cheb
        self.cheb_schedule = None
        if self.use_cheb:
            if num_epochs is None:
                raise ValueError(
                    "can't produce chebs without num epochs info being passed in"
                )
            self.cheb_schedule = get_chebs(num_epochs)

        self.total_iterations = num_epochs * num_batches_per_epoch
        if not self.total_iterations:
            raise ValueError(
                "missing total iterations, which is calced from num epochs and num iters per epoch param"
            )

        # lr
        self.starting_lr = lr
        self.current_lr = lr

        # warmup - we'll use default recommended in Ma/Yarats unless user specifies num iterations
        # -=-=-=-=-=-=-=-=-=-=-=-=-=--=-=--=-=-
        self.use_warmup = use_warmup
        self.warmup_complete = False
        self.warmup_type = warmup_type
        self.warmup_pct_default = warmup_pct_default

        if num_warmup_iterations is None:
            beta_warmup_iters = math.ceil(
                (2 / (1 - betas[1]))
            )  # default untuned linear warmup

            beta_pct = beta_warmup_iters / self.total_iterations
            # print(f"beta_warmup_pct = {beta_pct}")

            # this can be unreasonable for short runs...so let's compare vs warmup pct % of total epochs
            if beta_pct > 0.45:
                warmup_auto_pct = int(self.warmup_pct_default * self.total_iterations)
                self.num_warmup_iters = warmup_auto_pct
            else:
                self.num_warmup_iters = beta_warmup_iters

        else:  # user passed in specific num
            self.num_warmup_iters = num_warmup_iterations

        # warm down
        self.min_lr = warmdown_min_lr
        self.warmdown_lr_delta = self.starting_lr - self.min_lr
        self.warmdown_active = warmdown_active

        if self.warmdown_active:
            self.warm_down_start_pct = warmdown_start_pct
            self.start_warm_down = int(
                self.warm_down_start_pct * num_epochs * num_batches_per_epoch
            )
            self.warmdown_total_iterations = (
                self.total_iterations - self.start_warm_down
            )
            self.warmdown_displayed = False  # print when warmdown begins...
            self.warmup_curr_pct = 0.01  # used to verify warmup reaches full set point.

            """
            print(f"debug warmdown:\n")
            print(f"warm_down_start_pct = {self.warm_down_start_pct}")
            print(f"num_epochs = {self.num_epochs}, num_batches per epoch = {self.num_batches_per_epoch}")
            print(f" start warmdown at {self.start_warm_down}")
            print(f" total iterations of warmdown = {self.warmdown_total_iterations}")
            print(f" total lr delta = {self.warmdown_lr_delta}")
            """

        self.current_epoch = 0
        self.current_iter = 0

        self.use_gc = using_gc
        self.use_gcnorm = using_normgc
        self.gc_conv_only = gc_conv_only

        # epochs
        self.epoch_count = 0

        # momentum
        self.momentum_pnm = momentum_type == "pnm"

        self.pnm_momentum = pnm_momentum_factor

        # decay
        self.decay = weight_decay
        self.decay_type = decay_type
        self.param_size = 0

        # logging - need to update things before moving these 2 into self.logging toggle
        self.cheb_logging = []
        self.tracking_lr = []

        if self.logging:
            self.tracking_variance_sum = []
            self.tracking_variance_normalized = []

        # display
        engine = "AdamW" if not self.use_madgrad else "MadGrad"

        # print out initial settings to make usage easier

        self.show_settings()

    def __setstate__(self, state):
        super().__setstate__(state)

    # show settings at init or if called
    def show_schedule(self):
        if not self.tracking_lr:
            print(
                "No data from training yet.  Please train and then use this to show the lr curves"
            )
            return
        x = self.tracking_lr
        plt.plot(x)
        maxlr = max(x)
        minlr = min(x)
        startlr = x[0]
        plt.title(
            f"Ranger21 learning rate schedule\nStart={startlr:.2E}\nMax ={maxlr:.2E}\n,Min={minlr:.2E}\n"
        )
        plt.show()

    def show_settings(self):
        print(f"Ranger21 optimizer ready with following settings:\n")
        print(f"Core optimizer = {self.core_engine}")
        print(f"Learning rate of {self.starting_lr}\n")

        print(
            f"Important - num_epochs of training = ** {self.num_epochs} epochs **\nplease confirm this is correct or warmup and warmdown will be off\n"
        )

        if self.use_adabelief:
            print(f"using AdaBelief for variance computation")
        if self.use_warmup:
            print(
                f"Warm-up: {self.warmup_type} warmup, over {self.num_warmup_iters} iterations\n"
            )
        if self.lookahead_active:
            print(
                f"Lookahead active, merging every {self.lookahead_mergetime} steps, with blend factor of {self.lookahead_alpha}"
            )
        if self.normloss_active:
            print(f"Norm Loss active, factor = {self.normloss_factor}")
        if self.decay:
            print(f"Stable weight decay of {self.decay}")

        if self.use_gc:
            print(f"Gradient Centralization = On\n")
        else:
            print("Gradient Centralization = Off\n")

        print(f"Adaptive Gradient Clipping = {self.agc_active}")
        if self.agc_active:
            print(f"\tclipping value of {self.agc_clip_val}")
            print(f"\tsteps for clipping = {self.agc_eps}")

        if self.warmdown_active:
            print(
                f"\nWarm-down: Linear warmdown, starting at {self.warm_down_start_pct*100}%, iteration {self.start_warm_down} of {self.total_iterations}"
            )
            print(f"warm down will decay until {self.min_lr} lr")

    # lookahead functions
    def clear_cache(self):
        """clears the lookahead cached params """

        print(f"clearing lookahead cache...")
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                try:
                    la_params = param_state["lookahead_params"]
                except:
                    print(f"no lookahead cache present.")
                    return

                if len(la_params):
                    param_state["lookahead_params"] = torch.zeros_like(p.data)
        print(f"lookahead cache cleared")

    def clear_and_load_backup(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                p.data.copy_(param_state["backup_params"])
                del param_state["backup_params"]

    def backup_and_load_cache(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["backup_params"] = torch.zeros_like(p.data)
                param_state["backup_params"].copy_(p.data)
                p.data.copy_(param_state["lookahead_params"])

    def unit_norm(self, x):
        """ axis-based Euclidean norm"""
        # verify shape
        keepdim = True
        dim = None

        xlen = len(x.shape)
        # print(f"xlen = {xlen}")

        if xlen <= 1:
            keepdim = False
        elif xlen in (2, 3):  # linear layers
            dim = 1
        elif xlen == 4:  # conv kernels
            dim = (1, 2, 3)
        else:
            dim = tuple(
                [x for x in range(1, xlen)]
            )  # create 1,..., xlen-1 tuple, while avoiding last dim ...

        return x.norm(dim=dim, keepdim=keepdim, p=2.0)

    def agc(self, p):
        """clip gradient values in excess of the unitwise norm.
        the hardcoded 1e-6 is simple stop from div by zero and no relation to standard optimizer eps
        """

        # params = [p for p in parameters if p.grad is not None]
        # if not params:
        #    return

        # for p in params:
        p_norm = self.unit_norm(p).clamp_(self.agc_eps)
        g_norm = self.unit_norm(p.grad)

        max_norm = p_norm * self.agc_clip_val

        clipped_grad = p.grad * (max_norm / g_norm.clamp(min=1e-6))

        new_grads = torch.where(g_norm > max_norm, clipped_grad, p.grad)
        p.grad.detach().copy_(new_grads)

    def warmup_dampening(self, lr, step):

        style = self.warmup_type
        warmup = self.num_warmup_iters

        if style is None:
            return lr

        if step > warmup:
            if not self.warmup_complete:

                if not self.warmup_curr_pct == 1.0:
                    print(
                        f"Error - lr did not achieve full set point from warmup, currently {self.warmup_curr_pct}"
                    )

                self.warmup_complete = True
                print(f"\n** Ranger21 update = Warmup complete - lr set to {lr}\n")
            return lr

        if style == "linear":
            self.warmup_curr_pct = min(1.0, (step / warmup))
            new_lr = lr * self.warmup_curr_pct
            self.current_lr = new_lr
            return new_lr

        # elif style == "exponential":
        # return lr * (1.0 - math.exp(-step / warmup))

        else:
            raise ValueError(f"warmup type {style} not implemented.")

    def get_warm_down(self, lr, iteration):
        """ linear style warmdown """
        if iteration < self.start_warm_down:
            return lr

        if iteration > self.start_warm_down - 1:
            # print when starting
            if not self.warmdown_displayed:
                print(
                    f"\n** Ranger21 update: Warmdown starting now.  Current iteration = {iteration}....\n"
                )
                self.warmdown_displayed = True

            warmdown_iteration = (
                iteration + 1
            ) - self.start_warm_down  # to force the first iteration to be 1 instead of 0

            if warmdown_iteration < 1:
                print(
                    f" warning - iteration started at {iteration} and {self.start_warm_down} with value {warmdown_iteration}"
                )
                warmdown_iteration = 1
            # print(f"warmdown iteration = {warmdown_iteration}")
            # linear start 3672  5650 total iterations 1972 iterations

            warmdown_pct = warmdown_iteration / (
                self.warmdown_total_iterations + 1
            )  # +1 to offset that we have to include first as an iteration to support 1 index instead of 0 based.
            if warmdown_pct > 1.00:
                print(f"error in warmdown pct calc.  new pct = {warmdown_pct}")
                print(f"auto handled but please report issue")
                warmdown_pct = 1.00

            # .5
            lr_range = self.warmdown_lr_delta

            reduction = lr_range * warmdown_pct
            # print(f"lr reduction = {reduction} for {warmdown_pct} with iter {warmdown_iteration} and total iter {iteration}")
            new_lr = self.starting_lr - reduction
            if new_lr < self.min_lr:
                print(f"error in warmdown - lr below min lr. current lr = {new_lr}")
                print(f"auto handling but please report issue!")
                new_lr = self.min_lr

            self.current_lr = new_lr
            return new_lr

            # new_lr = (
            #    self.min_lr
            #    + self.starting_lr
            #    * (1 + math.cos(math.pi * warmdown_iteration / self.warmdown_total_iterations))
            #    / 2
            # )
            # self.current_lr = new_lr
            # return new_lr

    # def new_epoch_handler(self, iteration):

    #    self.epoch_count +=1

    def track_epochs(self, iteration):
        self.current_iter += 1
        if self.current_iter % self.num_batches_per_epoch == 0:
            self.current_iter = 0
            self.epoch_count += 1
            # print(f"New epoch, current epoch = {self.epoch_count}")
            self.tracking_lr.append(self.current_lr)

            # load lookup params for validation
            if self.lookahead_active and self.lookahead_validation_load:
                self.backup_and_load_cache()

    def get_cheb_lr(self, lr, iteration):

        # first confirm we are done with warmup
        if self.use_warmup:
            if iteration < self.num_warmup_iters + 1:
                return lr

        # compute epoch
        current_epoch = (iteration // self.num_batches) + 1
        # print(f"current epoch for cheb = {current_epoch}")
        self.current_epoch = current_epoch
        index = current_epoch - 2
        if index < 0:
            index = 0
        if index > len(self.cheb_schedule) - 1:
            index = len(self.cheb_schedule) - 1

        cheb_value = self.cheb_schedule[index]

        if self.cheb_logging[:-1] != cheb_value:
            self.cheb_logging.append(cheb_value)

        return lr * cheb_value

    def get_variance(self):
        return self.tracking_variance_sum

    def get_state_values(self, group, state):
        beta1, beta2 = group["betas"]
        mean_avg = state["mean_avg"]
        variance_avg = state["variance_avg"]

        return beta1, beta2, mean_avg, variance_avg

    # @staticmethod
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None and isinstance(closure, collections.abc.Callable):
            with torch.enable_grad():
                loss = closure()

        param_size = 0
        variance_ma_sum = 0.0

        # phase 1 - accumulate all of the variance_ma_sum to use in stable weight decay

        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                # if not self.param_size:
                param_size += p.numel()

                # apply agc if enabled
                if self.agc_active:
                    self.agc(p)

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("sparse matrix not supported atm")

                state = self.state[p]
                momentum = group["momentum"]

                # State initialization
                if len(state) == 0:
                    # print("init state")
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["grad_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["variance_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    if self.lookahead_active:
                        state["lookahead_params"] = torch.zeros_like(p.data)
                        state["lookahead_params"].copy_(p.data)

                    if self.use_adabelief:
                        state["variance_ma_belief"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    if self.momentum_pnm:
                        state["neg_grad_ma"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_variance_ma"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    # Cumulative products of beta1
                    # state["beta1_prod"] = torch.ones_like(
                    #    p.data, memory_format=torch.preserve_format
                    # )

                # centralize gradients
                if self.use_gc:
                    grad = centralize_gradient(
                        grad,
                        gc_conv_only=self.gc_conv_only,
                    )
                if self.use_gcnorm:
                    grad = normalize_gradient(grad)
                # else:
                #    grad = uncentralized_grad

                # phase 1, variance computations

                state["step"] += 1

                step = state["step"]
                lr = group["lr"]

                beta1, beta2 = group["betas"]
                grad_ma = state["grad_ma"]

                bias_correction2 = 1 - beta2 ** state["step"]
                # print(f"bias2 = {bias_correction2}")

                variance_ma = state["variance_ma"]
                if self.use_adabelief:
                    variance_ma_belief = state["variance_ma_belief"]

                # print(f"variance_ma, upper loop = {variance_ma}")

                # update the exp averages
                if self.use_adabelief:
                    grad_ma.mul_(beta1).add_(grad, alpha=1 - beta1)
                    grad_residual = grad - grad_ma
                    variance_ma_belief.mul_(beta2).addcmul(
                        grad_residual, grad_residual, value=1 - beta2
                    )
                # print(f"upper loop grad = {grad.shape}")
                variance_ma.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # print(f"variance_ma, grad adjusted")
                variance_ma_debiased = variance_ma / bias_correction2

                variance_ma_sum += variance_ma_debiased.sum()
                # print(f"variance_ma_sum = {variance_ma_sum}")
                # else: #madgrad

                # should we dupe variance_ma since stable is assuming adam style] variance?

                # stable wd
                # variance_ma_sum += grad_sum_sq.sum()

            # print(f"variance hat sum = {exp_avg_sq_hat_sum}")
            # Calculate the sqrt of the mean of all elements in exp_avg_sq_hat

            # we will run this first epoch only and then memoize
        if not self.param_size:
            self.param_size = param_size
            print(f"params size saved")
            print(f"total param groups = {i+1}")
            print(f"total params in groups = {j+1}")

        if not self.param_size:
            raise ValueError("failed to set param size")

        # stable weight decay
        if self.use_madgrad:
            variance_normalized = torch.pow(variance_ma_sum / param_size, 1/3)
        else:
            variance_normalized = math.sqrt(variance_ma_sum / param_size)
        # variance_mean = variance_ma_sum / param_size
        if math.isnan(variance_normalized):
            raise RuntimeError("hit nan for variance_normalized")

        # debugging/logging
        if self.logging:
            self.tracking_variance_sum.append(variance_ma_sum.item())
            self.tracking_variance_normalized.append(variance_normalized)

        # print(f"variance_mean = {variance_mean}")
        # print(f"variance_normalized = {variance_normalized}")
        # else:
        #    variance_normalized = math.pow((variance_ma / self.param_size), .3333)

        # print(f"variance mean sqrt = {variance_normalized}")

        # phase 2 - apply weight decay and step
        # ===========================================
        for group in self.param_groups:
            # print(f"In second phase loop")
            step = state["step"]

            # Perform stable weight decay
            decay = group["weight_decay"]
            eps = group["eps"]
            lr = group["lr"]
            momentum = group["momentum"]

            beta1, beta2 = group["betas"]

            # warmup
            # ======================
            if self.use_warmup and not self.warmup_complete:
                lr = self.warmup_dampening(lr, step)
                # print(f"lr = {lr}")

            # chebyshev
            # ===================
            if self.use_cheb and self.warmup_complete:
                lr = self.get_cheb_lr(lr, step)

            # warmdown
            # ==========
            if self.warmdown_active:
                orig_lr = lr
                lr = self.get_warm_down(lr, step)
                assert lr > 0, "lr went negative"

            # madgrad outer
            if self.use_madgrad:
                ck = 1 - momentum
                lamb = lr * math.pow(step, 0.5)

            # stable decay and / or norm loss
            # ==================================
            if decay:
                if not self.use_madgrad:
                    # stable weight decay
                    p.data.mul_(1 - decay * lr / variance_normalized)
                else:
                    p.data.mul_(1 - decay * lamb / variance_normalized)

            if self.normloss_active:
                # apply norm loss
                unorm = self.unit_norm(p.data)
                correction = (
                    2 * self.normloss_factor * (1 - torch.div(1, unorm + self.eps))
                )
                p.mul_(1 - lr * correction)

            # innner loop, params
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                inner_grad = p.grad

                if self.use_madgrad:
                    # ================== madgrad ============================
                    if "grad_sum_sq" not in state:
                        state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                        state["s"] = torch.zeros_like(p.data).detach()
                        if momentum != 0:
                            state["x0"] = torch.clone(p.data).detach()

                    if momentum != 0.0 and grad.is_sparse:
                        raise RuntimeError(
                            "momentum != 0 is not compatible with sparse gradients"
                        )

                    # centralize gradients
                    if self.use_gc:
                        inner_grad = centralize_gradient(
                            inner_grad,
                            gc_conv_only=self.gc_conv_only,
                        )

                    grad_sum_sq = state["grad_sum_sq"]
                    s = state["s"]
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3)
                        if self.softplus:
                            rms = F.softplus(rms, beta=self.beta_softplus)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    # Accumulate second moments

                    # print(f" grad = {grad}")
                    # print(f"lamb = {lamb}")
                    # print(f"gsumsq = {grad_sum_sq}")

                    grad_sum_sq.addcmul_(inner_grad, inner_grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3)
                    if self.softplus:
                        rms = F.softplus(rms, beta=self.beta_softplus)

                    # Update s
                    s.data.add_(inner_grad, alpha=lamb)

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)

                else:  # adam with pnm core
                    # ============= adamW with pnm option ========================

                    grad = p.grad

                    beta1, beta2 = group["betas"]

                    grad_ma = state["grad_ma"]
                    variance_ma = state["variance_ma"]
                    if self.use_adabelief:
                        variance_ma_belief = state["variance_ma_belief"]

                    if self.momentum_pnm:

                        max_variance_ma = state["max_variance_ma"]

                        if state["step"] % 2 == 1:
                            grad_ma, neg_grad_ma = (
                                state["grad_ma"],
                                state["neg_grad_ma"],
                            )
                        else:
                            grad_ma, neg_grad_ma = (
                                state["neg_grad_ma"],
                                state["grad_ma"],
                            )

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    if self.momentum_pnm:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_variance_ma, variance_ma, out=variance_ma)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (variance_ma.sqrt() / math.sqrt(bias_correction2)).add_(
                            group["eps"]
                        )

                    # centralize gradients
                    if self.use_gc:
                        grad = centralize_gradient(
                            grad,
                            gc_conv_only=self.gc_conv_only,
                        )
                    if self.use_gcnorm:
                        grad = normalize_gradient(grad)

                    if not self.use_adabelief:
                        grad_ma.mul_(beta1 ** 2).add_(grad, alpha=1 - beta1 ** 2)

                    noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)

                    step_size = lr / bias_correction1

                    # softplus the denom
                    if self.softplus:
                        denom = F.softplus(denom, beta=self.beta_softplus)

                    pnmomentum = (
                        grad_ma.mul(1 + self.momentum_pnm)
                        .add(neg_grad_ma, alpha=-self.momentum_pnm)
                        .mul(1 / noise_norm)
                    )

                    p.addcdiv_(pnmomentum, denom, value=-step_size)

                    # denom = variance_biased_ma.sqrt().add(eps)

                    # step_size = lr / bias_correction1

                    # update weights
                    # p.data.add_(weight_mod, alpha=-step_size)
                    # p.addcdiv_(grad_ma, denom, value=-step_size)
        # print(f"\n End optimizer step\n")

        # end of step processes....

        # lookahead
        # ---------------------
        if self.lookahead_active:
            self.lookahead_process_step()

        self.track_epochs(step)
        return loss

    #   Lookahead merge process
    def lookahead_process_step(self):
        """handles blending of params for lookahead step"""

        if not self.lookahead_active:
            return
        self.lookahead_step += 1

        if self.lookahead_step >= self.lookahead_mergetime:
            self.lookahead_step = 0
            # merge lookahead cached params and save current ones
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]

                    p.data.mul_(self.lookahead_alpha).add_(
                        param_state["lookahead_params"],
                        alpha=1.0 - self.lookahead_alpha,
                    )
                    # save for next merge
                    param_state["lookahead_params"].copy_(p.data)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups