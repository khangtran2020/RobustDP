import torch

def fgsm_attack(image:torch.Tensor, rad:torch.Tensor, data_grad:torch.Tensor):
    batch_size = image.size(dim=0)
    sign_data_grad = data_grad.sign()
    perturbed_image = image + rad.unsqueeze(dim=1).repeat(1, int(image.numel() / batch_size)).view(image.size())*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(image:torch.Tensor, label:torch.Tensor, steps:int, model:torch.nn.Module, rad:torch.Tensor, alpha:float, device:torch.device, random:bool=False):
    
    model.to(device)
    img = image.detach().clone().to(device)
    lab = label.detach().clone().to(device)
    obj = torch.nn.CrossEntropyLoss()

    adv = img.clone()
    bsz = img.size(dim=0)

    if random:
        delta = torch.empty_like(adv).normal_(mean=0, std=rad)
        delta = delta * min(1, rad / (delta.norm(p=2).item() + 1e-12))
        adv = adv + delta
        adv = torch.clamp(adv, min=0, max=1).detach()
        del delta

    for i in range(steps):

        adv.requires_grad = True
        pred = model(adv)
        loss = obj(pred, lab)

        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        adv = adv.detach() + alpha * grad.sign()
        delta = adv - img
        delta = delta * torch.clamp(rad / (torch.flatten(delta, start_dim=1).norm(p=2, dim=1) + 1e-12), max=1.0).unsqueeze(dim=1).repeat(1, int(img.numel() / bsz)).view(img.size())
        adv = torch.clamp(img + delta, min=0, max=1).detach()

    return adv