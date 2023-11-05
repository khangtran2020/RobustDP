import sys
import torch
from Utils.console import console

def clipping_weight(model:torch.nn.Module, clip:float, mode:str='clean', lay_out_size:list=None):
    # if mode == 'dp': console.log(vars(model))
    with torch.no_grad():
        i = 1
        for n, p in model.named_parameters():
            if ('weight' in n):
                if 'cnn_layers' in n:
                    if mode == 'clean':
                        norm = p.norm(p=2) * model.lay_out_size[f'conv_{i}']
                    else:
                        norm = p.norm(p=2) * lay_out_size[f'conv_{i}']
                    i += 1
                else:
                    norm = p.norm(p=2)
                p.data = p * min(1, clip / (norm + 1e-12))

    return model

def check_clipped(model:torch.nn.Module, clip:float, mode:str='clean', lay_out_size:list=None):

    res = True
    with torch.no_grad():
        i = 1
        for n, p in model.named_parameters():
            if ('weight' in n):
                if 'cnn_layers' in n:
                    if mode == 'clean':
                        cond = (p.norm(p=2) * model.lay_out_size[f'conv_{i}'] - clip).abs().item() > 1e-5
                    else:
                        cond = (p.norm(p=2) * lay_out_size[f'conv_{i}'] - clip).abs().item() > 1e-5
                    i += 1
                else:
                    cond = (p.norm(p=2) - clip).abs().item() > 1e-5
                
                if cond:
                    console.log(f'[bold][red] Failed initial clip check :x:: clip is {clip} and norm is {p.norm(p=2).item()}')
                    sys.exit()
    if res:
        console.log('[bold][green] Pass initial clip check: :white_check_mark:')
    return res



