import torch
import torchmetrics
from copy import deepcopy
from rich.progress import Progress
from typing import Dict
from Models.modules.training_utils import EarlyStopping
from Models.utils import lip_clip, clip_weight
from Utils.console import console
from Utils.tracking import tracker_log, wandb

def traindp(args, tr_loader:torch.utils.data.DataLoader, va_loader:torch.utils.data.DataLoader, model:torch.nn.Module, 
            device:torch.device, history:Dict, name=str):
    
    model_name = '{}.pt'.format(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.num_class > 1:
        objective = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        pred_fn = torch.nn.Softmax(dim=1).to(device)
        metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
    else:
        objective = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
        pred_fn = torch.nn.Sigmoid().to(device)
        metrics = torchmetrics.classification.BinaryAccuracy().to(device)

    console.log(f"[green]Train / Optimizing model with optimizer[/green]: {optimizer}")
    console.log(f"[green]Train / Objective of the training process[/green]: {objective}")
    console.log(f"[green]Train / Predictive activation[/green]: {pred_fn}")
    console.log(f"[green]Train / Evaluating with metrics[/green]: {metrics}")

    model.to(device)
    model.train()

    # model = clipping_weight(model=model, clip=args.clipw, mode=args.gen_mode, lay_out_size=lay_out_size)

    console.log(f"Using sigma={args.ns} and C={args.clip}")
    model_list = []

    # es = EarlyStopping(patience=15, mode='max', verbose=False)

    with Progress(console=console) as progress:

        tk_tr = progress.add_task("[red]Training...", total=args.epochs)
        tk_up = progress.add_task("[green]Updating...", total=len(tr_loader))
        tk_ev = progress.add_task("[cyan]Evaluating...", total=len(va_loader))

        # progress.reset(task_id=task1)
        for epoch in range(args.epochs):

            tr_loss = 0
            ntr = 0
            
            # train
            model.train()
            num_step = len(tr_loader)

            for bi, d in enumerate(tr_loader):
                model.zero_grad()
                optimizer.zero_grad()
                data, target = d
                data = data.to(device)
                target = target.to(device)
                num_data = data.size(dim=0)
                
                if (epoch == args.epochs - 1) & (bi == num_step - 1):
                    
                    num_data_mini = int(num_data / args.num_mo)

                    for mit in range(args.num_mo):
                        model.load_state_dict(torch.load(args.model_path + model_name))
                        model.zero_grad()
                        optimizer.zero_grad()

                        mini_data = data[mit*num_data_mini:(mit + 1)*num_data_mini].clone()
                        mini_targ = target[mit*num_data_mini:(mit + 1)*num_data_mini].clone()

                        pred = model(mini_data)
                        loss = objective(mini_data, mini_targ)
                        saved_var = dict()
                        for tensor_name, tensor in model.named_parameters():
                            saved_var[tensor_name] = torch.zeros_like(tensor).to(device)

                        for li, los in enumerate(loss):
                            los.backward(retain_graph=True)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                            for tensor_name, tensor in model.named_parameters():
                                if tensor.grad is not None:
                                    new_grad = tensor.grad
                                    saved_var[tensor_name].add_(new_grad)
                            model.zero_grad()

                        for tensor_name, tensor in model.named_parameters():
                            if tensor.grad is not None:
                                saved_var[tensor_name].add_(torch.FloatTensor(tensor.grad.shape).normal_(0, args.ns * args.clip).to(device))
                                tensor.grad = saved_var[tensor_name] / num_data_mini
                        optimizer.step()
                        model_list.append(deepcopy(model))

                        pred = pred_fn(pred).detach()
                        metrics.update(pred, mini_targ)
                        tr_loss += loss.detach().mean().item()*num_data_mini
                        ntr += num_data_mini

                else:

                    pred = model(data)
                    loss = objective(data, target)

                    saved_var = dict()
                    for tensor_name, tensor in model.named_parameters():
                        saved_var[tensor_name] = torch.zeros_like(tensor).to(device)

                    for li, los in enumerate(loss):
                        los.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        for tensor_name, tensor in model.named_parameters():
                            if tensor.grad is not None:
                                new_grad = tensor.grad
                                saved_var[tensor_name].add_(new_grad)
                        model.zero_grad()

                    for tensor_name, tensor in model.named_parameters():
                        if tensor.grad is not None:
                            saved_var[tensor_name].add_(torch.FloatTensor(tensor.grad.shape).normal_(0, args.ns * args.clip).to(device))
                            tensor.grad = saved_var[tensor_name] / num_data

                    optimizer.step()
                    pred = pred_fn(pred).detach()
                    metrics.update(pred, target)
                    model = clip_weight(model=model, clip=args.clipw)
                    tr_loss += loss.detach().mean().item()*num_data
                    ntr += num_data
                progress.advance(tk_up)

            tr_loss = tr_loss / ntr 
            tr_perf = metrics.compute().item()
            metrics.reset()   
           

            va_loss = 0
            nva = 0

            # validation
            if (epoch == args.epochs - 1):
                console.log(f"# of model: {len(model_list)}")
                for i, m in enumerate(model_list):
                    torch.save(m.cpu().state_dict(), args.model_path + f"model_{i+1}_{model_name}")
                    m.eval()

                with torch.no_grad():

                    for bi, d in enumerate(va_loader):
                        data, target = d
                        data = data.to(device)
                        target = target.to(device)

                        preds = None
                        for miv, m in enumerate(model_list):
                            if miv == 0:
                                preds = m(data)
                            else:
                                preds = preds + m(data)
                        
                        preds = preds / len(model_list)

                        loss = objective(preds, target)
                        preds = pred_fn(preds)
                        metrics.update(preds, target)
                        va_loss += loss.mean().item()*preds.size(dim=0)
                        nva += preds.size(dim=0)
                        progress.advance(tk_ev)
            else:
                model.eval()
                with torch.no_grad():
                    for bi, d in enumerate(va_loader):
                        data, target = d
                        data = data.to(device)
                        target = target.to(device)
                        pred = model(data)
                        loss = objective(pred, target)
                        pred = pred_fn(pred)
                        metrics.update(pred, target)
                        va_loss += loss.mean().item()*pred.size(dim=0)
                        nva += pred.size(dim=0)
                        progress.advance(tk_ev)

                va_loss = va_loss / nva 
                va_perf = metrics.compute().item()
                metrics.reset()
                torch.save(model.state_dict(), args.model_path + model_name)

            results = {
                "Target epoch": epoch+1,
                "Target train/loss": tr_loss, 
                "Target train/acc": tr_perf, 
                "Target val/loss": va_loss, 
                "Target val/acc": va_perf,
            }
            history['tr_loss'].append(tr_loss)
            history['tr_perf'].append(tr_perf)
            history['va_loss'].append(va_loss)
            history['va_perf'].append(va_perf)
            # es(epoch=epoch, epoch_score=va_perf, model=model, model_path=args.model_path + model_name)
            tracker_log(dct=results)

            progress.console.print(f"Epoch {epoch}: [yellow]loss[/yellow]: {tr_loss}, [yellow]acc[/yellow]: {tr_perf}, [yellow]va_loss[/yellow]: {va_loss}, [yellow]va_acc[/yellow]: {va_perf}") 
            progress.advance(tk_tr)
            progress.reset(tk_up)
            progress.reset(tk_ev)
        console.log(f"Done Training target model: :white_check_mark:")
    return model_list, history

def evaltdp(args, te_loader:torch.utils.data.DataLoader, model_list:list, device:torch.device, history:Dict):
    
    if args.num_class > 1:
        objective = torch.nn.CrossEntropyLoss().to(device)
        pred_fn = torch.nn.Softmax(dim=1).to(device)
        metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
    else:
        objective = torch.nn.BCEWithLogitsLoss().to(device)
        pred_fn = torch.nn.Sigmoid().to(device)
        metrics = torchmetrics.classification.BinaryAccuracy().to(device)
    
    console.log(f"[green]Evaluate Test / Objective of the training process[/green]: {objective}")
    console.log(f"[green]Evaluate Test / Predictive activation[/green]: {pred_fn}")
    console.log(f"[green]Evaluate Test / Evaluating with metrics[/green]: {metrics}")
    for m in model_list:
        m.to(device)
        m.eval()


    with Progress(console=console) as progress:
        task1 = progress.add_task("[red]Evaluating Test ...", total=len(te_loader))
        te_loss = 0
        nte = 0
        # validation
        
        with torch.no_grad():
            for bi, d in enumerate(te_loader):
                data, target = d
                data = data.to(device)
                target = target.to(device)
                
                for mi, m in enumerate(model_list):
                    if mi == 0:
                        preds = m(data)
                    else:
                        preds = preds + m(data)
                loss = objective(preds, target)
                preds = pred_fn(preds)
                metrics.update(preds, target)
                te_loss += loss.item()*preds.size(dim=0)
                nte += preds.size(dim=0)
                progress.update(task1, advance=bi+1)

            te_loss = te_loss / nte 
            te_perf = metrics.compute().item()
            wandb.run.summary['te_loss'] = '{0:.3f}'.format(te_loss)
            wandb.run.summary['te_acc'] = '{0:.3f}'.format(te_perf)
            history['best_test_loss'] = te_loss
            history['best_test_perf'] = te_perf
            metrics.reset()

    return history
