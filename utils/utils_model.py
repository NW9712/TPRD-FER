import torch

def get_model(cfg, clip = None):
    if 'TPRD_baseline' in cfg.model and clip:
        from models.TPRD_baseline import TPRD_baseline
        model = TPRD_baseline(clip, cfg)
    elif 'TPRD_disentangle' in cfg.model and clip:
        from models.TPRD_disentangle import TPRD_disentangle
        model = TPRD_disentangle(clip, cfg)
    else:
        raise NotImplementedError

    if isinstance(cfg.device, list):
        device_id=list(cfg.device)
        model=torch.nn.DataParallel(model, device_ids=device_id).to(device_id[0])
    else:
        from utils.utils_device import get_device
        model=model.to(get_device(cfg))
    return model