import torch
import torchmetrics
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Dict
from rich.progress import Progress
from Utils.console import console
from Utils.tracking import wandb

def fgsm_attack(image:torch.Tensor, r:float, data_grad:torch.Tensor):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + r*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def denorm(batch:torch.Tensor, device:torch.device, mean=[0.1307], std=[0.3081]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def robust_eval_clean(args, model:torch.nn.Module, device:torch.device, te_loader:torch.utils.data.DataLoader, num_plot:int, history:Dict):

    with console.status("Evaluating robustness") as status:
        # Accuracy counter
        correct = 0
        adv_examples = []
        metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
        # Loop over all examples in test set

        for i, batch in enumerate(te_loader):

            data, target = batch
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            output = model(data)
            init_pred = output.max(dim=1)

            top_2, _ = torch.topk(input=output, k=2)
            L = args.clipw ** model.num_trans
            M = args.clipw
            radius = (top_2[:,0] - top_2[:,1]).abs().squeeze() / (L*M*args.img_sz**2)
            
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            data_denorm = denorm(data, device=device)

            adv_data = torch.Tensor([]).to(device)
            for j in range(data.size(dim=0)):
                perturbed_data = fgsm_attack(data_denorm[j], radius[j], data_grad[j])
                perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
                adv_data = torch.cat((adv_data, perturbed_data_normalized), dim=0)
            output = model(perturbed_data_normalized)

            # Check for success
            final_pred = output.max(dim=1)
            metrics.update(final_pred, init_pred)

            if (i == 0):
                org_img = data[:num_plot]
                adv_img = adv_data[:num_plot]
                log_test_predictions(org_imgs=org_img, adv_imgs=adv_img, labels=target[:num_plot], org_pred=init_pred[:num_plot], adv_pred=final_pred[:num_plot])

        # Calculate final accuracy for this epsilon
        final_acc = metrics.compute().item()
        history['correctness_of_bound'] = final_acc
        console.log(f"Corretness of bound performance: {final_acc}")
        console.log(f'[bold][green]Done Evaluating robustness: :white_check_mark:')

def log_test_predictions(org_imgs, adv_imgs, labels, org_pred, adv_pred):

    columns=["id", "org_img", "adv_img", "label", "original prediction", "adversarial prediction"]
    test_table = wandb.Table(columns=columns)

    log_org_images = org_imgs.cpu().numpy()
    log_adv_images = adv_imgs.cpu().numpy()
    log_labels = labels.cpu().numpy()
    log_org_preds = org_pred.cpu().numpy()
    log_adv_preds = adv_pred.cpu().numpy()

    # adding ids based on the order of the images
    idx = 0
    for lab, org_im, org_pr, adv_im, adv_pr in zip(log_labels, log_org_images, log_org_preds, log_adv_images, log_adv_preds):
        img_id = f'Image {idx}'
        test_table.add_data(img_id, lab, wandb.Image(org_im), org_pr, wandb.Image(adv_im), adv_pr)
        idx += 1
    wandb.log({"Prediction" : test_table})
