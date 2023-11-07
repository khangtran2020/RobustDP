import sys
import torch
from Utils.console import console

def lip_clip(model:torch.nn.Module, clip:float):

    with torch.no_grad():
        i = 1
        for n, p in model.named_parameters():
            if ('weight' in n) & ('last_lay' not in n):
                if 'cnn_layers' in n:
                    k = p.size(dim=-1)
                    c1 = p.size(dim=1)
                    c0 = p.size(dim=0)
                    norm_1 = p.detach().view(int(c0*k), int(c1*k)).norm(p=2).item()
                    norm_2 = p.detach().view(int(c0*k), int(c1*k)).norm(p=2).item()
                    norm_3 = p.detach().view(int(c0), int(c1*k*k)).norm(p=2).item()
                    norm_4 = p.detach().view(int(c0*k*k), int(c1)).norm(p=2).item()
                    norm = k * min([norm_1, norm_2, norm_3, norm_4])
                else:
                    norm = p.norm(p=2)
                p.data = p.data / (norm + 1e-12)

    return model

def clip_weight(model:torch.nn.Module, clip:float):
    norm = model.last_lay.weight.data.norm(p=2)
    model.last_lay.weight.data = model.last_lay.weight.data / (norm + 1e-12)
    return model

def check_clipped(model:torch.nn.Module, clip:float):

    res = True
    with torch.no_grad():
        i = 1
        for n, p in model.named_parameters():
            if ('weight' in n):
                if 'cnn_layers' in n:
                    k = p.size(dim=-1)
                    c1 = p.size(dim=1)
                    c0 = p.size(dim=0)
                    norm_1 = p.detach().view(int(c0*k), int(c1*k)).norm(p=2).item()
                    norm_2 = p.detach().view(int(c0*k), int(c1*k)).norm(p=2).item()
                    norm_3 = p.detach().view(int(c0), int(c1*k*k)).norm(p=2).item()
                    norm_4 = p.detach().view(int(c0*k*k), int(c1)).norm(p=2).item()
                    norm = k * min([norm_1, norm_2, norm_3, norm_4])
                    cond = (norm - clip) > 1e-5
                else:
                    cond = (p.norm(p=2) - clip).abs().item() > 1e-5
                
                if cond:
                    console.log(f'[bold][red] Failed initial clip check :x:: clip is {clip} and norm is {p.norm(p=2).item()}')
                    sys.exit()
    if res:
        console.log('[bold][green] Pass initial clip check: :white_check_mark:')
    return res



