"""
The script is used to configure learning rate scheduler and optimizer similar to Swin transformer
"""

import torch
from torch import optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay


def configure_parameters(model, skip_list=(), skip_keywords=(), lr=1e-5, lr_scale=10.0, pretrained=False):
    has_decay_bb = []
    no_decay_bb = []
    # backbone or not
    has_decay_not_bb = []
    no_decay_not_bb = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            if pretrained and "vit_encoder" in name:
                no_decay_bb.append(param)
            else:
                no_decay_not_bb.append(param)
            # print(f"{name} has no weight decay")
        else:
            if pretrained and "vit_encoder" in name:
                has_decay_bb.append(param)
            else:
                has_decay_not_bb.append(param)
    return [{'params': has_decay_bb, 'lr': lr/lr_scale},
            {'params': has_decay_not_bb, 'lr': lr},
            {'params': no_decay_bb, 'weight_decay': 0., 'lr': lr/lr_scale},
            {'params': no_decay_not_bb, 'weight_decay': 0., 'lr': lr}
            ]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


def init_optimizer(cfg, model):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = configure_parameters(model, skip, skip_keywords,
                                      lr=cfg.TRAIN.BASE_LR,
                                      lr_scale=cfg.BACKBONE.PRETRAIN_LR_SCALE,
                                      pretrained=True)
    opt_lower = cfg.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=cfg.TRAIN.OPTIMIZER.EPS, betas=cfg.TRAIN.OPTIMIZER.BETAS,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    return optimizer

# steal from swin transformer
def init_scheduler(cfg, optimizer, n_iter_per_epoch):
    num_steps = int(cfg.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(cfg.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(cfg.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if cfg.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=cfg.TRAIN.MIN_LR,
            warmup_lr_init=cfg.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif cfg.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=cfg.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif cfg.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=cfg.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=cfg.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler
