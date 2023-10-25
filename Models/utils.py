import torch

def clipping_weight(model:torch.nn.Module, clip:float):

    with torch.no_grad():
        for n, p in model.named_parameters():
            if ('weight' in n):
                p = p * min(1, clip / (p.norm(p=2) + 1e-12))

    return model

def check_clipped(model:torch.nn.Module, clip:float):

    res = True
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ('weight' in n) & (p.norm(p=2).item() > clip):
                res = False
                break

    return res



