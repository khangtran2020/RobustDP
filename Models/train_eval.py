import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from typing import Sequence
from Models.forward import forward_clean, forward_dpsgd
Device = torch.device

# train 

def tr_clean(loader:DataLoader, model:Module, obj:Module, opt:Optimizer, 
             metric:Metric, pred_fn:Module, clipw:float, device:torch.device):
    
    model.to(device)
    model.train()

    tr_loss = 0
    num_data = 0

    for bi, batch in enumerate(loader):
        
        model.zero_grad()
        opt.zero_grad()
        loss, n = forward_clean(model=model, batch=batch, device=device, metric=metric, 
                                obj=obj, opt=opt, pred_fn=pred_fn)
        tr_loss += loss * n
        num_data += n

    tr_loss = tr_loss / num_data
    tr_perf = metric.compute()
    metric.reset()

    return tr_loss, tr_perf.item()

def tr_dpsgd(loader:DataLoader, model:Module, obj:Module, opt:Optimizer, 
             metric:Metric, pred_fn:Module, device:torch.device, clipw:float, clip:float, ns:float, get:bool=False):
    
    model.to(device)
    model.train()

    batch = next(iter(loader))    
    
    model.zero_grad()
    if get == False:
        tr_loss, n = forward_dpsgd(model=model, batch=batch, device=device, metric=metric, 
                                   obj=obj, opt=opt, pred_fn=pred_fn, clipw=clipw, clip=clip, ns=ns)

        tr_perf = metric.compute()
        metric.reset()

        return tr_loss, tr_perf.item()
    else:

        las_lay, n = forward_dpsgd(model=model, batch=batch, device=device, metric=metric, 
                                   obj=obj, opt=opt, pred_fn=pred_fn, clipw=clipw, clip=clip, ns=ns, get=get)
        return las_lay, n
        

def eval_fn(loader:DataLoader, model:Module, obj:Module, metric:Metric, clipw:float, device:torch.device, pred_fn:Module):
    model.to(device)
    avg_loss = 0
    num_pt = 0
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            feat, target = batch
            feat = feat.to(device)
            target = target.to(device, dtype=torch.long)
            score = model(feat)
            loss = obj(score, target).mean()
            avg_loss += loss.item()*feat.size(dim=0)
            num_pt += feat.size(dim=0)
            pred = pred_fn(score)
            metric.update(pred, target.int())

    avg_loss = avg_loss / num_pt
    perf = metric.compute()
    metric.reset()

    return avg_loss, perf.item()

def eval_multi_fn(loader:DataLoader, models:Sequence[Module], obj:Module, metric:Metric, device:Device, pred_fn:Module):

    for model in models:
        model.to(device)
        model.eval()

    avg_loss = 0
    num_data = 0

    with torch.no_grad():

        for bi, batch in enumerate(loader):
            feat, target= batch
            feat = feat.to(device)
            target = target.to(device, dtype=torch.long)
            for j, model in enumerate(models):
                if j == 0:
                    score = model(feat)
                else:
                    score = score + model(feat)
            score = score/len(models)
            loss = obj(score, target).mean()
            avg_loss += loss.item() * feat.size(dim=0)
            num_data += feat.size(dim=0)
            pred = pred_fn(score)
            metric.update(pred, target.int())
        
        perf = metric.compute().item()
        metric.reset()
    return avg_loss / num_data, perf

