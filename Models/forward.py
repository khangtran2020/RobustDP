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
    target = target.long().to(device, dtype=torch.float)
    score = model(feat)
    loss = obj(score, target)
    loss.backward()
    opt.step()
    pred = pred_fn(score.detach())
    metric.update(pred, target.int())

    return loss.item(), feat.size(dim=0)

def forward_dpsgd(model:Module, batch:tuple, device:Device, metric:Metric, opt: Optimizer, 
                  obj:Module, pred_fn:Module, clip:float, ns:float, l2:torch.Tensor=None):
    
    feat, target, _ = batch
    feat = feat.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    score = model(feat)
    score = torch.squeeze(score)
    loss = obj(score, target)
    num_pt = feat.size(dim=0)

    saved_var = dict()
    for tensor_name, tensor in model.named_parameters():
        saved_var[tensor_name] = torch.zeros_like(tensor).to(device)

    for pos, j in enumerate(loss):
        j.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for tensor_name, tensor in model.named_parameters():
            if tensor.grad is not None:
                new_grad = tensor.grad.clone()
                saved_var[tensor_name].add_(new_grad)
        model.zero_grad()

    for tensor_name, tensor in model.named_parameters():
        if tensor.grad is not None:
            saved_var[tensor_name].add_(torch.FloatTensor(tensor.grad.shape).normal_(0, ns * clip ).to(device))
            tensor.grad = saved_var[tensor_name] / num_pt    

    if l2 is not None:
        l2.backward()

    opt.step()
    pred = pred_fn(score.detach())
    metric.update(pred, target.int())

    return loss.mean().item(), feat.size(dim=0)

def forward_fairdp(batch:tuple, model:Module, device: Device, opt:Optimizer, 
                   obj:Module, clip:float, ns:float, pred_fn:Module, get:bool=False):

    feat, target, _ = batch
    feat = feat.to(device)
    target = target.to(device)

    opt.zero_grad()
    model.zero_grad()

    score = model(feat)
    score = torch.squeeze(score, dim=-1)
    loss = obj(score, target)
    num_pt = feat.size(dim=0)    

    if get == False:

        saved_var = dict()
        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name] = torch.zeros_like(tensor).to(device)

        for pos, j in enumerate(loss):
            j.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for tensor_name, tensor in model.named_parameters():
                if tensor.grad is not None:
                    new_grad = tensor.grad
                    saved_var[tensor_name].add_(new_grad)
            model.zero_grad()

        for tensor_name, tensor in model.named_parameters():
            if tensor.grad is not None:
                saved_var[tensor_name].add_(
                    torch.FloatTensor(tensor.grad.shape).normal_(0, clip*ns).to(device))
                tensor.grad = saved_var[tensor_name] / num_pt

        opt.step()
        return loss.mean().item(), num_pt
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
                    if 'out_layer' not in tensor_name:
                        new_grad = tensor.grad.clone()
                        saved_var[tensor_name].add_(new_grad)
                    else:
                        last_lay.append(tensor.grad)
            model.zero_grad()

        for name, tensor in model.named_parameters():
            if tensor.grad is not None:
                if 'out_layer' not in tensor_name:
                    saved_var[name].add_(
                        torch.FloatTensor(tensor.grad.shape).normal_(0, clip * ns).to(device))
                    tensor.grad = saved_var[name] / num_pt
                else:
                    tensor.grad = torch.zeros_like(tensor).to(device)

        opt.step()
        # console.log(f"Last layer grad: {last_lay}")
        return last_lay, num_pt
