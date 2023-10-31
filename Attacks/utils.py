import os
import random
import string
import torch
import wandb
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from Attacks.attacks import fgsm_attack, pgd_attack
from typing import Dict
from Utils.console import console

def robust_eval_clean(args, model:torch.nn.Module, device:torch.device, te_loader:torch.utils.data.DataLoader, num_plot:int, history:Dict):

    with console.status("Evaluating robustness") as status:
        # Accuracy counter
        correct = 0
        adv_examples = []
        metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
        metrics_tar = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
        # Loop over all examples in test set
        las_w = model.last_lay.weight.data.clone().detach()
        num_c = args.num_class


        for i, batch in enumerate(te_loader):

            data, target = batch
            data, target = data.to(device), target.to(device)
            if args.att_mode.split('-')[0] == 'fgsm':
                data.requires_grad = True
            org_scores = model(data)
            top_k, index = torch.topk(input=org_scores, k=num_c)
            wei = las_w[index]
            L = args.clipw ** model.num_trans # lipschitz condition
            for i in range(1, num_c):
                M = (wei[:, 0] - wei[:, i]).norm(p=2, dim=1)
                rad = (top_k[:,0] - top_k[:,i]).abs().squeeze() / L*M
                # if args.debug:
                    # print(f"# pt {data.size(dim=0)}, size of M: {M.size()}, size of rad: {rad}")
                if i == 1:
                    radius = rad.clone()
                else:
                    radius = torch.min(radius, rad)

            init_pred = org_scores.max(1, keepdim=True)[1]
            data_denorm = denorm(data, device=device)

            if args.att_mode.split('-')[0] == 'fgsm':
                loss = torch.nn.CrossEntropyLoss()(org_scores, target)
                model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                adv_data = fgsm_attack(data_denorm, radius / args.img_sz**2, data_grad)
                adv_data = transforms.Normalize((0.1307,), (0.3081,))(adv_data)
            elif args.att_mode.split('-')[0] == 'pgd':
                adv_data = pgd_attack(image=data_denorm, label=target, steps=args.pgd_steps, model=model, rad=radius, alpha=2/255, device=device)

            adv_scores = model(adv_data)
            final_pred = adv_scores.max(1, keepdim=True)[1]
            metrics.update(final_pred, init_pred)
            metrics_tar.update(torch.nn.Softmax(dim=1)(adv_scores), target)

            if (i == 0):
                org_img = data[:num_plot]
                org_scr = org_scores[:num_plot]
                org_prd = init_pred[:num_plot]

                adv_img = adv_data[:num_plot]
                adv_scr = adv_scores[:num_plot]
                adv_prd = final_pred[:num_plot]
                labels = target[:num_plot]
                rads = radius[:num_plot]

                log_test_predictions(org_imgs=org_img, org_scr=org_scr, org_prd=org_prd, adv_img=adv_img, adv_scr=adv_scr, 
                                     adv_prd=adv_prd,labels=labels, radius=rads, name=f"Predictions under {args.att_mode.split('-')[0]} attack")

        # Calculate final accuracy for this epsilon
        final_acc = metrics.compute().item()
        certified_acc = metrics_tar.compute().item()
        history['correctness_of_bound'] = final_acc
        history['certified_acc'] = certified_acc
        console.log(f"Corretness of bound performance: {final_acc}")
        console.log(f"Certified Accuracy: {certified_acc}")
        wandb.summary[f"Corretness of bound performance"] = f"{final_acc}"
        wandb.summary[f"Certified Accuracy"] = f"{certified_acc}"
        console.log(f'[bold][green]Done Evaluating robustness: :white_check_mark:')

def log_test_predictions(org_img:torch.Tensor, org_scr:torch.Tensor, org_prd:torch.Tensor, 
                         adv_img:torch.Tensor, adv_scr:torch.Tensor, adv_prd:torch.Tensor,
                         labels:torch.Tensor, radius:torch.Tensor, name:str):

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
        og_sc = draw_score(score=log_org_scr[i])
        og_pr = log_org_prd[i]

        ad_im = wandb.Image(log_adv_img[i])
        ad_sc = draw_score(score=log_adv_scr[i])
        ad_pr = log_adv_prd[i]

        test_table.add_data(img_id, lab, rad, og_im, og_sc, og_pr, ad_im, ad_sc, ad_pr)
        idx += 1
    wandb.log({name : test_table})

def denorm(batch:torch.Tensor, device:torch.device, mean=[0.1307], std=[0.3081]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def draw_score(score:np.ndarray):

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
    plt.savefig(path)
    img = Image.open(path)
    arr = np.array(img)
    return arr
