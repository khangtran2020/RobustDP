import torch
import torchmetrics
from copy import deepcopy
from rich.progress import Progress
from typing import Dict
from Models.utils import init_model
from Models.modules.adam import CustomAdamOptimizer
from Models.train_eval import tr_dpsgd, eval_fn, eval_multi_fn
from Utils.console import console
from Utils.tracking import tracker_log, wandb

def traindp(args, tr_loader:torch.utils.data.DataLoader, va_loader:torch.utils.data.DataLoader, model:torch.nn.Module, 
            device:torch.device, history:Dict, name=str):
    
    model_name = '{}.pt'.format(name)
    param_dct = {}
    for n, p in model.named_parameters():
        param_dct[n] = p.data.clone()
    optimizer = CustomAdamOptimizer(params=param_dct, lr=args.lr, device=device)


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
        tk_ev = progress.add_task("[cyan]Evaluating...", total=len(va_loader))

        num_step = len(tr_loader)
        # progress.reset(task_id=task1)
        for epoch in range(args.epochs):
            
            if epoch < args.epochs - 1:
                tr_loss, tr_perf = tr_dpsgd(loader=tr_loader, model=model, obj=objective, opt=optimizer, metric=metrics, pred_fn=pred_fn, clip=args.clip, ns=args.ns, device=device)
                va_loss, va_perf = eval_fn(loader=va_loader, model=model, obj=objective, metric=metrics, clipw=None, pred_fn=pred_fn, device=device)
                torch.save(model.state_dict(), args.model_path + model_name)
            else:
                grad, bz = tr_dpsgd(loader=tr_loader, model=model, obj=objective, opt=optimizer, metric=metrics, pred_fn=pred_fn, clip=args.clip, ns=args.ns, device=device, get=True)
                state_dict = model.state_dict()
                with torch.no_grad():

                    model_list = []
                    mini_batch = int(bz / args.num_mo)

                    for i in range(args.num_mo):
                        model_temp = init_model(args=args)
                        model_temp.load_state_dict(state_dict=state_dict)

                        saved_var = {}
                        for tensor_name, tensor in model_temp.named_parameters():
                            if 'last_lay' in tensor_name: 
                                saved_var[tensor_name] = torch.zeros_like(tensor).to(device)

                        for pos, j in enumerate(grad[i * mini_batch:(i + 1) * mini_batch]):
                            for tensor_name, tensor in model_temp.named_parameters():
                                if tensor.grad is not None:
                                    if 'last_lay' in tensor_name:
                                        saved_var[tensor_name].add_(j)

                        for tensor_name, tensor in model_temp.named_parameters():
                            if tensor.grad is not None:
                                if 'last_lay' in tensor_name:
                                    saved_var[tensor_name].add_(
                                        torch.FloatTensor(tensor.grad.shape).normal_(0, args.clip*args.ns).to(device))
                                    tensor.grad = saved_var[tensor_name]
                                    tensor.data = tensor.data - args.lr * tensor.grad
                        model_list.append(model_temp)

                    tr_loss, tr_perf = eval_multi_fn(loader=tr_loader, models=model_list, obj=objective, metric=metrics, device=device, pred_fn=pred_fn)
                    va_loss, va_perf = eval_multi_fn(loader=va_loader, models=model_list, obj=objective, metric=metrics, device=device, pred_fn=pred_fn)
                    for i, m in enumerate(model_list):
                        torch.save(m.state_dict(), args.model_path + f'model_{i}_{name}.pt')

            # scheduler.step(metrics=va_loss)

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
                target = target.to(device, dtype=torch.long)
                
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
