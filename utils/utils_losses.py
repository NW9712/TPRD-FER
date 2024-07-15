import torch
import numpy as np

def get_loss(name, cfg):

    if name == 'CrossEntropyLoss':
        if 'ces_weight' in cfg.keys():
            class_weights = torch.FloatTensor(np.array(cfg.ces_weight)).cuda()
            return torch.nn.CrossEntropyLoss(class_weights, reduction='mean')
        else:
            return torch.nn.CrossEntropyLoss(reduction='mean')
    if name == 'Soft_CrossEntropyLoss':
        from utils.losses.losses import Soft_CrossEntropyLoss
        return Soft_CrossEntropyLoss(num_classes = 8, reduction='mean')
    elif name == 'WeakSup_ConLoss':
        from utils.losses.losses import WeakSupConLoss
        if cfg.temperature:
            return WeakSupConLoss(temperature=cfg.temperature)
        else: #default
            return WeakSupConLoss()
    elif name == 'Sup_ConLoss' or name == 'NCELoss':
        from utils.losses.losses import SupConLoss
        if cfg.temperature:
            return SupConLoss(temperature=cfg.temperature, base_temperature=cfg.temperature)
        else: #default
            return SupConLoss()
    elif name == 'kl_Loss':
        from utils.losses.losses import kl_loss_compute
        return kl_loss_compute
    elif name == 'js_loss':
        from utils.losses.losses import js_loss_compute
        return js_loss_compute
    elif name == 'refining_loss':
        from utils.losses.losses import refining_loss_compute
        return refining_loss_compute
    elif name =='LabelSmoothing_CrossEntropy':
        from utils.losses.losses import LabelSmoothing_CrossEntropy
        return LabelSmoothing_CrossEntropy()
    else:
        raise NotImplementedError


