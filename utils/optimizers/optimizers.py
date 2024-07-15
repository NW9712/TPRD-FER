import torch

def SGD(parameters,cfg):
    optimizer = torch.optim.SGD(parameters, lr=cfg.initial_lr, weight_decay=cfg.weight_decay,momentum=cfg.momentum)
    return optimizer

def Adam(parameters,cfg):
    optimizer = torch.optim.Adam(parameters, lr=cfg.initial_lr, weight_decay=cfg.weight_decay)
    return optimizer

def AdamW(parameters,cfg):
    optimizer = torch.optim.AdamW(parameters, lr=cfg.initial_lr, weight_decay=cfg.weight_decay)
    return optimizer
