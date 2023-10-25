import torch
import torchmetrics
from rich.progress import Progress
from typing import Dict
from Models.modules import EarlyStopping
from Models.utils import clipping_weight
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from Utils.console import console
from Utils.tracking import tracker_log, wandb

def traindp(args, tr_loader:torch.utils.data.DataLoader, va_loader:torch.utils.data.DataLoader, model:torch.nn.Module, device:torch.device, history:Dict, name=str):
    
    model_name = '{}.pt'.format(name)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    if args.num_class > 1:
        objective = torch.nn.CrossEntropyLoss().to(device)
        pred_fn = torch.nn.Softmax(dim=1).to(device)
        metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
    else:
        objective = torch.nn.BCEWithLogitsLoss().to(device)
        pred_fn = torch.nn.Sigmoid().to(device)
        metrics = torchmetrics.classification.BinaryAccuracy().to(device)

    console.log(f"[green]Train / Optimizing model with optimizer[/green]: {optimizer}")
    console.log(f"[green]Train / Objective of the training process[/green]: {objective}")
    console.log(f"[green]Train / Predictive activation[/green]: {pred_fn}")
    console.log(f"[green]Train / Evaluating with metrics[/green]: {metrics}")

    model.to(device)
    model.train()

    privacy_engine = PrivacyEngine()
    model, optimizer, tr_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=tr_loader,
        epochs=args.epochs,
        target_epsilon=args.eps,
        target_delta=1e-5,
        max_grad_norm=args.clip,
    )

    console.log(f"Using sigma={optimizer.noise_multiplier} and C={args.clip}")

    es = EarlyStopping(patience=15, mode='max', verbose=False)

    with Progress(console=console) as progress:

        tk_tr = progress.add_task("[red]Training...", total=args.epochs)
        tk_up = progress.add_task("[green]Updating...", total=len(tr_loader))
        tk_ev = progress.add_task("[cyan]Evaluating...", total=len(va_loader))

        # progress.reset(task_id=task1)
        for epoch in range(args.epochs):

            tr_loss = 0
            ntr = 0
            num_step = len(tr_loader)

            # train
            model.train()
            with BatchMemoryManager(
                    data_loader=tr_loader, 
                    max_physical_batch_size=args.max_bs, 
                    optimizer=optimizer
                ) as memory_safe_data_loader:
                for bi, d in enumerate(memory_safe_data_loader):
                    model.zero_grad()
                    optimizer.zero_grad()
                    data, target = d
                    pred = model(data)
                    loss = objective(pred, target)
                    pred = pred_fn(pred)
                    metrics.update(pred, target)
                    loss.backward()
                    optimizer.step()
                    model = clipping_weight(model=model, clip=args.clipw)
                    tr_loss += loss.item()*pred.size(dim=0)
                    ntr += pred.size(dim=0)
                    progress.advance(tk_up)

            tr_loss = tr_loss / ntr 
            tr_perf = metrics.compute().item()
            metrics.reset()   

            va_loss = 0
            nva = 0

            # validation
            model.eval()
            with torch.no_grad():
                for bi, d in enumerate(va_loader):
                    data, target = d
                    pred = model(data)
                    loss = objective(pred, target)
                    pred = pred_fn(pred)
                    metrics.update(pred, target)
                    va_loss += loss.item()*pred.size(dim=0)
                    nva += pred.size(dim=0)
                    progress.advance(tk_ev)

            va_loss = va_loss / nva 
            va_perf = metrics.compute().item()
            metrics.reset()

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
            es(epoch=epoch, epoch_score=va_perf, model=model, model_path=args.model_path + model_name)
            tracker_log(dct=results)

            progress.console.print(f"Epoch {epoch}: [yellow]loss[/yellow]: {tr_loss}, [yellow]acc[/yellow]: {tr_perf}, [yellow]va_loss[/yellow]: {va_loss}, [yellow]va_acc[/yellow]: {va_perf}") 
            progress.advance(tk_tr)
            progress.reset(tk_up)
            progress.reset(tk_ev)
        console.log(f"Done Training target model: :white_check_mark:")
    model.load_state_dict(torch.load(args.model_path + model_name))
    return model, history