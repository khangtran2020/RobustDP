import os
import math
import random
import string
import torch
import wandb
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from typing import Dict
from PIL import Image
from Attacks.attacks import fgsm_attack, pgd_attack, pgd_attack_dp
from Utils.console import console
from Utils.utils import get_index_by_value

def robust_eval_clean(args, model:torch.nn.Module, device:torch.device, te_loader:torch.utils.data.DataLoader, num_plot:int, history:Dict):

    with console.status("Evaluating robustness") as status:

        model.eval()
        # Accuracy counter
        metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
        metrics_tar = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
        # Loop over all examples in test set
        las_w = model.last_lay.weight.data.clone().detach()
        console.log(f"Norm weight of the last layer: {las_w}, with size {las_w.size()}")
        # check_clipped(model=model, clip=1.0)
        num_c = args.num_class

        pred = torch.Tensor([]).to(device)
        gtar = torch.Tensor([]).to(device)
        crad = torch.Tensor([]).to(device)
        for i, batch in enumerate(te_loader):

            data, target = batch
            data, target = data.to(device), target.to(device)
            if args.att_mode.split('-')[0] == 'fgsm':
                data.requires_grad = True
            org_scores = model(data)
            top_k, idx = torch.topk(input=org_scores, k=num_c)
            wei = las_w[idx]
            # console.log(f"weight diff size: {wei.size()}, since the idx has size: {idx.size}")
            for j in range(1, num_c):
                M = (wei[:, 0, :] - wei[:, j, :]).norm(p=2, dim=1)
                rad = (top_k[:,0] - top_k[:,j]).abs().squeeze() / (M * math.sqrt(2))
                # console.log(f"weight diff size: {M.size()}, score diff size: {(top_k[:,0] - top_k[:,j]).size()}")
                if j == 1:
                    radius = rad.clone()
                else:
                    radius = torch.min(radius, rad)

            init_pred = org_scores.max(1, keepdim=True)[1]
            if args.data == 'mnist':
                data_denorm = denorm(data, device=device)
            elif args.data == 'cifar10':
                data_denorm = denorm(data, device=device, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            elif args.data == 'utk':
                data_denorm = data 

            if args.att_mode.split('-')[0] == 'fgsm':
                loss = torch.nn.CrossEntropyLoss()(org_scores, target)
                model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                adv_data = fgsm_attack(data_denorm, radius / args.img_sz**2, data_grad)
                adv_data = transforms.Normalize((0.1307,), (0.3081,))(adv_data)
            elif args.att_mode.split('-')[0] == 'pgd':
                adv_data = pgd_attack(image=data_denorm, label=target, steps=args.pgd_steps, model=model, rad=radius, alpha=2/255, device=device)
                if args.data == 'mnist':
                    adv_data = transforms.Normalize((0.1307,), (0.3081,))(adv_data)
                elif args.data == 'cifar10':
                    adv_data = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(adv_data)

            adv_scores = model(adv_data)
            final_pred = adv_scores.max(1, keepdim=True)[1]
            metrics.update(final_pred, init_pred)
            metrics_tar.update(torch.nn.Softmax(dim=1)(adv_scores), target)

            # console.log(f"Radius size: {radius.size()}, init pred size: {init_pred.size()}, target size: {target.size()}")
            crad = torch.cat((crad, radius), dim=0)
            pred = torch.cat((pred, init_pred.squeeze()), dim=0)
            gtar = torch.cat((gtar, target), dim=0)

            if (i == 0):

                if args.data == 'mnist':
                    org_img = data[:num_plot]
                    org_scr = org_scores[:num_plot]
                    org_prd = init_pred[:num_plot]

                    adv_img = adv_data[:num_plot]
                    adv_scr = adv_scores[:num_plot]
                    adv_prd = final_pred[:num_plot]
                elif args.data == 'cifar10':
                    org_img = data[:num_plot].permute(0, 2, 3, 1)
                    org_scr = org_scores[:num_plot]
                    org_prd = init_pred[:num_plot]

                    adv_img = adv_data[:num_plot].permute(0, 2, 3, 1)
                    adv_scr = adv_scores[:num_plot]
                    adv_prd = final_pred[:num_plot]
                elif args.data == 'utk':
                    org_img = data[:num_plot]
                    org_scr = org_scores[:num_plot]
                    org_prd = init_pred[:num_plot]

                    adv_img = adv_data[:num_plot]
                    adv_scr = adv_scores[:num_plot]
                    adv_prd = final_pred[:num_plot]
                labels = target[:num_plot]
                rads = radius[:num_plot]

                print(f"Logging test prediction")
                log_test_predictions(org_img=org_img, org_scr=org_scr, org_prd=org_prd, adv_img=adv_img, adv_scr=adv_scr, 
                                     adv_prd=adv_prd,labels=labels, radius=rads, name=f"Predictions under {args.att_mode.split('-')[0]} attack", num_class=args.num_class)

        # Calculate final accuracy for this epsilon
        console.log(f"Radius size: {crad.size()}, init pred size: {pred.size()}, target size: {gtar.size()}")

        final_acc = metrics.compute().item()
        correct = (pred.int() == gtar.int()).int()
        rad_ls, cert_acc, img_arr = certified_accuracy(radius=crad, correct=correct)

        images = wandb.Image(
            img_arr, caption="Certified Accuracy"
        )
            
        wandb.log({"Certified Accuracy": images})

        history['correctness_of_bound'] = final_acc
        history['certified_radius'] = rad_ls
        history['certified_acc'] = cert_acc
        console.log(f"Corretness of bound performance: {final_acc}")
        console.log(f"Certified Radius: {rad_ls}")
        console.log(f"Certified Accuracy: {cert_acc}")
        wandb.summary[f"Corretness of bound performance"] = f"{final_acc}"
        wandb.summary[f"Certified Accuracy"] = f"{cert_acc}"
        wandb.summary[f"Certified Radius"] = f"{rad_ls}"
        console.log(f'[bold][green]Done Evaluating robustness: :white_check_mark:')

def log_test_predictions(org_img:torch.Tensor, org_scr:torch.Tensor, org_prd:torch.Tensor, 
                         adv_img:torch.Tensor, adv_scr:torch.Tensor, adv_prd:torch.Tensor,
                         labels:torch.Tensor, radius:torch.Tensor, name:str, num_class:int):

    columns=["id", "label", "radius", "original image", "original score", "original prediction", "adversarial image", "adversarial score", "adversarial prediction"]
    test_table = wandb.Table(columns=columns)

    batch_size = org_img.size(dim=0)

    log_org_img = org_img.detach().cpu().numpy()
    log_org_scr = org_scr.detach().cpu().numpy()
    log_org_prd = org_prd.detach().cpu().numpy()

    log_adv_img = adv_img.detach().cpu().numpy()
    log_adv_scr = adv_scr.detach().cpu().numpy()
    log_adv_prd = adv_prd.detach().cpu().numpy()

    log_lab = labels.detach().cpu().numpy()
    log_rad = radius.detach().cpu().numpy()

    # adding ids based on the order of the images
    idx = 0
    for i in range(batch_size):
        img_id = f'Image {idx}'
        lab = log_lab[i]
        rad = log_rad[i]

        og_im = wandb.Image(log_org_img[i])
        og_sc = wandb.Image(draw_score(score=log_org_scr[i], num_lab=num_class))
        og_pr = log_org_prd[i]

        ad_im = wandb.Image(log_adv_img[i])
        ad_sc = wandb.Image(draw_score(score=log_adv_scr[i], num_lab=num_class))
        ad_pr = log_adv_prd[i]

        # print(og_sc.shape, ad_sc.shape)
        test_table.add_data(img_id, lab, rad, og_im, og_sc, og_pr, ad_im, ad_sc, ad_pr)
        idx += 1
    wandb.run.log({name: test_table})   

def denorm(batch:torch.Tensor, device:torch.device, mean=[0.1307], std=[0.3081]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def draw_score(score:np.ndarray, num_lab:int):

    label = np.arange(len(score)).astype(np.int16)
    bar_label = [f'Label {i}' for i in label]
    name = ''.join(random.choice(string.ascii_lowercase) for i in range(20))
    path = f'results/dict/{name}.jpg'
    while (os.path.exists(path)):
        name = ''.join(random.choice(string.ascii_lowercase) for i in range(20))
        path = f'results/dict/{name}.jpg'

    plt.figure(figsize=(5,5))
    plt.bar(x=label, height=score)
    plt.ylabel('Score')
    plt.xticks(np.arange(num_lab))
    plt.savefig(path, dpi=200, bbox_inches='tight')
    img = Image.open(path)
    arr = np.array(img)
    return arr

def certified_accuracy(radius:torch.Tensor, correct:torch.Tensor, custom_rad:torch.Tensor=None):
    
    rad_min = radius.min().item()
    rad_max = radius.max().item()

    print(correct.unique(return_counts=True))

    if custom_rad is not None:
        considered_rad = custom_rad
    else:
        considered_rad = torch.linspace(start=0, end=rad_max, steps=10)
    
    cert_acc = []

    for rad in considered_rad:
        rad_mask = (radius > rad).int()
        corr_mask = torch.logical_and(rad_mask, correct).float()
        cert_acc.append(corr_mask.mean().item())
    
    radius_ls = considered_rad.tolist()
    name = ''.join(random.choice(string.ascii_lowercase) for i in range(20))
    path = f'results/dict/{name}.jpg'
    while (os.path.exists(path)):
        name = ''.join(random.choice(string.ascii_lowercase) for i in range(20))
        path = f'results/dict/{name}.jpg'

    plt.figure(figsize=(5,5))
    plt.plot(range(len(radius_ls)), cert_acc)
    plt.xticks(np.arange(len(radius_ls)), [f'{a:.2f}' for a in radius_ls])
    plt.ylabel('Certified Accuracy')
    plt.xlabel('Certified Radius')
    plt.savefig(path, dpi=200, bbox_inches='tight')

    img = Image.open(path)
    arr = np.array(img)

    return radius_ls, cert_acc, arr

def hoeffding_bound(nobs, alpha, bonferroni_hyp_n=1):
    return math.sqrt(math.log(bonferroni_hyp_n / alpha) / (2 * nobs))

def robust_eval_dp(args, model_list:list, device:torch.device, te_loader:torch.utils.data.DataLoader, num_plot:int, history:Dict):

    torch.cuda.empty_cache()

    with console.status("Evaluating robustness") as status:
        # Accuracy counter
        metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
        metrics_tar = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
        # Loop over all examples in test set
        # check_clipped(model=model, clip=1.0)

        for m in model_list:
            m.to(device)
            m.eval()

        num_c = args.num_class
        pred_fn = torch.nn.Softmax(dim=1) if num_c > 1 else torch.nn.Sigmoid()

        pred = torch.Tensor([])
        gtar = torch.Tensor([])
        crad = torch.Tensor([])
        bound = hoeffding_bound(nobs=args.num_mo, alpha=args.alpha, bonferroni_hyp_n=num_c)
        log_epoch = 0

        for i, batch in enumerate(te_loader):

            data, target = batch
            data, target = data.to(device), target.to(device)
            if args.att_mode.split('-')[0] == 'fgsm':
                data.requires_grad = True

            for mi, m in enumerate(model_list):
                score = m(data)
                if mi == 0:
                    soft_score = pred_fn(score)
                    org_scores = score.clone()
                else:
                    soft_score = soft_score + pred_fn(score)
                    org_scores = org_scores + score.clone()

            org_scores = org_scores / args.num_mo
            soft_score = soft_score / args.num_mo
            
            top_k, idx = torch.topk(input=soft_score, k=2)
            
            lb = top_k[:, 0] - bound
            ub = top_k[:, 1] + bound
            abstain_mask = (lb > ub).int()
            idx = get_index_by_value(a=abstain_mask, val=1)
            radius = (lb - ub) / (4 * (num_c - 1) * args.clipw)
            init_pred = org_scores.max(1, keepdim=True)[1]


            if args.data == 'mnist':
                data_denorm = denorm(data, device=device)
            elif args.data == 'cifar10':
                data_denorm = denorm(data, device=device, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

            adv_data = pgd_attack_dp(image=data_denorm, label=target, steps=args.pgd_steps, model_list=model_list, rad=radius, alpha=2/255, device=device)
            if args.data == 'mnist':
                adv_data = transforms.Normalize((0.1307,), (0.3081,))(adv_data)
            elif args.data == 'cifar10':
                adv_data = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(adv_data)

            for mi, m in enumerate(model_list):
                score = m(adv_data)
                if mi == 0:
                    adv_scores = score.clone()
                else:
                    adv_scores = adv_scores + score.clone()

            adv_scores = adv_scores / args.num_mo
            final_pred = adv_scores.max(1, keepdim=True)[1]
            if len(idx) > 0:
                metrics.update(final_pred[idx], init_pred[idx])
            metrics_tar.update(torch.nn.Softmax(dim=1)(adv_scores), target)

            console.log(f"Radius size: {radius.size()}, init pred size: {init_pred.size()}, target size: {target.size()}")
            crad = torch.cat((crad, radius.cpu()), dim=0)
            pred = torch.cat((pred, init_pred.squeeze().cpu()), dim=0)
            gtar = torch.cat((gtar, target.cpu()), dim=0)

            if (i == log_epoch) & (len(idx) > 0):

                if args.data == 'mnist':
                    org_img = data[idx][:num_plot]
                    org_scr = org_scores[idx][:num_plot]
                    org_prd = init_pred[idx][:num_plot]

                    adv_img = adv_data[idx][:num_plot]
                    adv_scr = adv_scores[idx][:num_plot]
                    adv_prd = final_pred[idx][:num_plot]
                elif args.data == 'cifar10':
                    org_img = data[idx][:num_plot].permute(0, 2, 3, 1)
                    org_scr = org_scores[idx][:num_plot]
                    org_prd = init_pred[idx][:num_plot]

                    adv_img = adv_data[idx][:num_plot].permute(0, 2, 3, 1)
                    adv_scr = adv_scores[idx][:num_plot]
                    adv_prd = final_pred[idx][:num_plot]
                labels = target[idx][:num_plot]
                rads = radius[idx][:num_plot]

                print(f"Logging test prediction for epoch {log_epoch}, with length {len(idx)}")
                log_test_predictions(org_img=org_img, org_scr=org_scr, org_prd=org_prd, adv_img=adv_img, adv_scr=adv_scr, 
                                     adv_prd=adv_prd,labels=labels, radius=rads, name=f"Predictions under {args.att_mode.split('-')[0]} attack", num_class=args.num_class)
            else:
                log_epoch += 1
            del adv_data
            torch.cuda.empty_cache()

        # Calculate final accuracy for this epsilon
        console.log(f"Radius size: {crad.size()}, init pred size: {pred.size()}, target size: {gtar.size()}")

        final_acc = metrics.compute().item()
        correct = (pred.int() == gtar.int()).int()
        rad_ls, cert_acc, img_arr = certified_accuracy(radius=crad, correct=correct)

        images = wandb.Image(
            img_arr, caption="Certified Accuracy"
        )
            
        wandb.log({"Certified Accuracy": images})

        history['correctness_of_bound'] = final_acc
        history['certified_radius'] = rad_ls
        history['certified_acc'] = cert_acc
        console.log(f"Corretness of bound performance: {final_acc}")
        console.log(f"Certified Radius: {rad_ls}")
        console.log(f"Certified Accuracy: {cert_acc}")
        wandb.summary[f"Corretness of bound performance"] = f"{final_acc}"
        wandb.summary[f"Certified Accuracy"] = f"{cert_acc}"
        wandb.summary[f"Certified Radius"] = f"{rad_ls}"
        console.log(f'[bold][green]Done Evaluating robustness: :white_check_mark:')
