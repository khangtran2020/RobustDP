import torch
import numpy as np
from typing import Dict
from torchmetrics import Metric
from torch.optim import Optimizer
from torch.nn import Module
from Utils.console import console

Device = torch.device

def forward_clean(model:Module, batch:tuple, device:Device, metric:Metric, opt: Optimizer, 
                  obj:Module, pred_fn:Module):
    
    feat, target = batch
    feat = feat.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.long)
    score = model(feat)
    loss = obj(score, target)
    loss.backward()
    opt.step()
    pred = pred_fn(score.detach())
    metric.update(pred, target.int())

    return loss.item(), feat.size(dim=0)

def forward_dpsgd(model:Module, batch:tuple, device:Device, metric:Metric, opt: Optimizer, 
                  obj:Module, pred_fn:Module, clip:float, ns:float, l2:torch.Tensor=None, get:bool=False):
    
    feat, target = batch
    feat = feat.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.long)
    score = model(feat)
    loss = obj(score, target)
    num_pt = feat.size(dim=0)

    if get == False:
        saved_var = dict()
        for tensor_name, tensor in model.named_parameters():
            console.log(f"For {tensor_name} before: {tensor.norm(p=2)}")
            saved_var[tensor_name] = torch.zeros_like(tensor).to(device)

        for pos, j in enumerate(loss):
            j.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for tensor_name, tensor in model.named_parameters():
                if tensor.grad is not None:
                    new_grad = tensor.grad.clone()
                    saved_var[tensor_name].add_(new_grad)
            opt.zero_grad()

        for tensor_name, tensor in model.named_parameters():
            # if tensor.grad is not None:
            saved_var[tensor_name].add_(torch.FloatTensor(saved_var[tensor_name].shape).normal_(0, ns * clip ).to(device))
            tensor.grad = saved_var[tensor_name] / num_pt
        
        param_dct = {}
        for n, p in model.named_parameters():
            param_dct[n] = p
        
        new_param = opt.step(params=param_dct)
        for tensor_name, tensor in model.named_parameters():
            tensor.data = new_param[tensor_name].clone()

        pred = pred_fn(score.detach())
        metric.update(pred, target.int())
        model.zero_grad()
        return loss.mean().item(), feat.size(dim=0)
    else:
        last_lay = []
        saved_var = {}

        for name, tensor in model.named_parameters():
            saved_var[name] = torch.zeros_like(tensor).to(device)

        for pos, j in enumerate(loss):
            j.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for tensor_name, tensor in model.named_parameters():
                if tensor.grad is not None:
                    if 'last_lay' not in tensor_name:
                        new_grad = tensor.grad.clone()
                        saved_var[tensor_name].add_(new_grad)
                    else:
                        last_lay.append(tensor.grad)
            model.zero_grad()

        for name, tensor in model.named_parameters():
            if 'last_lay' not in tensor_name:
                saved_var[name].add_(
                    torch.FloatTensor(tensor.grad.shape).normal_(0, clip * ns).to(device))
                tensor.grad = saved_var[name] / num_pt
            else:
                tensor.grad = torch.zeros_like(tensor).to(device)

        param_dct = {}
        for n, p in model.named_parameters():
            param_dct[n] = p
        
        new_param = opt.step(params=param_dct)
        for tensor_name, tensor in model.named_parameters():
            tensor.data = new_param[tensor_name].clone()
        model.zero_grad()
        return last_lay, num_pt
