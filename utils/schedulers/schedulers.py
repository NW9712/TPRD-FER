import torch

def LambdaLR(optimizer,cfg):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = cfg.lr_Lambda_func)
    return scheduler

def StepLR(optimizer,cfg):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = cfg.lr_Step_size, gamma = cfg.lr_Step_gamma)
    return scheduler

def MultiStepLR(optimizer,cfg):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = cfg.lr_MultiStep_milestones, gamma = cfg.lr_MultiStep_gamma)
    return scheduler

def ExponentialLR(optimizer,cfg):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = cfg.lr_Exponential_gamma)
    return scheduler

def CosineAnnealingLR(optimizer,cfg):
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg.lr_CosineAnnealing_T_max, eta_min = cfg.lr_CosineAnnealing_eta_min)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg.lr_CosineAnnealing_T_max)
    return scheduler

def CosineAnnealingWarmRestarts(optimizer,cfg):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = cfg.lr_CosineAnnealingWarmRestarts_T_0, T_mult = cfg.lr_CosineAnnealingWarmRestarts_T_mult,eta_min = cfg.lr_CosineAnnealingWarmRestarts_eta_min)
    return scheduler

def ReduceLROnPlateau(optimizer,cfg):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = cfg.lr_ReduceLROnPlateau_mode, factor=cfg.lr_ReduceLROnPlateau_factor, patience=cfg.lr_ReduceLROnPlateau_patience,
                                                                    threshold = cfg.lr_ReduceLROnPlateau_threshold, threshold_mode=cfg.lr_ReduceLROnPlateau_threshold_mode, cooldown=cfg.lr_ReduceLROnPlateau_cooldown,
                                                                    min_lr=cfg.lr_ReduceLROnPlateau_min_lr, eps=cfg.lr_ReduceLROnPlateau_eps)
    return scheduler

