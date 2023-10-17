import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from Utils.console import console, log_table
from sklearn.model_selection import train_test_split
import numpy as np


def read_data(args, data_path='datasets/'):

    with console.status("Reading data ...") as status:
        # get train and test dataset
        if args.dataset == 'mnist':
            tr_dataset, te_dataset = get_mnist(path=data_path+args.dataset, size=args.img_size)
            target = tr_dataset.targets

        args.num_label = target.unique().size(dim=0)
        target = target.tolist()
        console.log(f"Finish fetching dataset: [green]{args.dataset}[/green]")
        dataset_size = len(tr_dataset)
        indices = list(range(dataset_size))
        id_tr, id_va, _, _ = train_test_split(indices, target, test_size=0.15, stratify=target)
        train_sampler = SubsetRandomSampler(id_tr)
        valid_sampler = SubsetRandomSampler(id_va)

        tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.bs, sampler=train_sampler, drop_last=True)
        va_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.bs, sampler=valid_sampler)
        te_loader = torch.utils.data.DataLoader(te_dataset, batch_size=args.bs)
        console.log(f"Finish generate dataloader")

        image, _ = next(iter(tr_loader))
        args.channel_in = image.size()[1].item()
        data_dict = {
            '# data train': f"{len(id_tr)}",
            '# data valid': f"{len(id_va)}",
            '# data test': f"{te_dataset.targets.size(dim=0)}",
            '# features':  f"{image.size()[1:].tolist()}",
            '# labels': f"{args.num_label}",
            'batch size': f"{image.size(dim=0)}"
        }
        log_table(dct=data_dict, name=f"{args.dataset}'s Property")
        console.log(f'[bold][green]Done!')

    return tr_loader, va_loader, te_loader


def get_mnist(path:str, size:int):
    
    train_dataset = torchvision.datasets.MNIST(f'{path}/', train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize((0.1307,), (0.3081,)),
                                                    transforms.Resize(size=size)
                                                    ])
                                                )

    test_dataset = torchvision.datasets.MNIST(f'{path}/', train=False, download=True, 
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        (0.1307,), (0.3081,)),
                                                    transforms.Resize(size=size)
                                                    ])
                                                )
    
    return train_dataset, test_dataset