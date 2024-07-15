import torch

def get_device(cfg):
    # if cfg['device'] is tuple:
    # if len(cfg.device) > 1:
    if isinstance(cfg.device, list):
        return torch.device('cuda:'+str(list(cfg['device'])[0])) if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device('cuda:'+str(cfg['device'])) if torch.cuda.is_available() else torch.device('cpu')