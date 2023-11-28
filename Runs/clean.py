import torch
import torchmetrics
from rich.progress import Progress
from typing import Dict
from Models.modules.training_utils import EarlyStopping
from Models.train_eval import tr_clean, eval_fn
from Utils.console import console
from Utils.tracking import tracker_log, wandb

def train(args, tr_loader:torch.utils.data.DataLoader, va_loader:torch.utils.data.DataLoader, model:torch.nn.Module, device:torch.device, history:Dict, name=str):
    
    model_name = '{}.pt'.format(name)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=10, threshold=0.0001, 
                                                           threshold_mode='rel',cooldown=0, min_lr=0, eps=1e-08)

    if args.num_class > 1:
        obj = torch.nn.CrossEntropyLoss().to(device)
        pred_fn = torch.nn.Softmax(dim=1).to(device)
        metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
    else:
        obj = torch.nn.BCEWithLogitsLoss().to(device)
        pred_fn = torch.nn.Sigmoid().to(device)
        metric = torchmetrics.classification.BinaryAccuracy().to(device)

    console.log(f"[green]Train / Optimizing model with optimizer[/green]: {optim}")
    console.log(f"[green]Train / Objective of the training process[/green]: {obj}")
    console.log(f"[green]Train / Predictive activation[/green]: {pred_fn}")
    console.log(f"[green]Train / Evaluating with metrics[/green]: {metric}")

    model.to(device)

    es = EarlyStopping(patience=15, mode='max', verbose=False)

    with Progress(console=console) as progress:

        tk_tr = progress.add_task("[red]Training...", total=args.epochs)

        # progress.reset(task_id=task1)
        for epoch in range(args.epochs):

            tr_loss, tr_perf = tr_clean(loader=tr_loader, model=model, obj=obj, opt=optim, metric=metric, clipw=args.clipw, pred_fn=pred_fn, device=device)
            va_loss, va_perf = eval_fn(loader=va_loader, model=model, obj=obj, metric=metric, clipw=args.clipw, pred_fn=pred_fn, device=device)
            scheduler.step(metrics=va_loss)

            results = {
                "Target train/loss": tr_loss, 
                "Target train/acc": tr_perf, 
                "Target val/loss": va_loss, 
                "Target val/acc": va_perf,
            }
            history['tr_loss'].append(tr_loss)
            history['tr_perf'].append(tr_perf)
            history['va_loss'].append(va_loss)
            history['va_perf'].append(va_perf)
            es(epoch=epoch, epoch_score=va_perf, model=model, model_path=args.model_path + model_name)
            tracker_log(dct=results)

            progress.console.print(f"Epoch {epoch}: [yellow]loss[/yellow]: {tr_loss}, [yellow]acc[/yellow]: {tr_perf}, [yellow]va_loss[/yellow]: {va_loss}, [yellow]va_acc[/yellow]: {va_perf}") 
            progress.advance(tk_tr)
        console.log(f"Done Training target model: :white_check_mark:")
    model.load_state_dict(torch.load(args.model_path + model_name))
    return model, history

def evalt(args, te_loader:torch.utils.data.DataLoader, model:torch.nn.Module, device:torch.device, history:Dict):

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
    model.to(device)
    with Progress(console=console) as progress:
        task1 = progress.add_task("[red]Evaluating Test ...", total=len(te_loader))
        te_loss = 0
        nte = 0
        # validation
        model.eval()
        with torch.no_grad():

            for bi, d in enumerate(te_loader):
                data, target = d
                data = data.to(device)
                target = target.to(device)
                pred = model(data)
                loss = objective(pred, target)
                pred = pred_fn(pred)
                metrics.update(pred, target)
                te_loss += loss.item()*pred.size(dim=0)
                nte += pred.size(dim=0)
                progress.update(task1, advance=bi+1)

            te_loss = te_loss / nte 
            te_perf = metrics.compute().item()
            wandb.run.summary['te_loss'] = '{0:.3f}'.format(te_loss)
            wandb.run.summary['te_acc'] = '{0:.3f}'.format(te_perf)
            history['best_test_loss'] = te_loss
            history['best_test_perf'] = te_perf
            metrics.reset()
    return history
