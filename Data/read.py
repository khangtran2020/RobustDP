import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from Utils.console import console


def read_data(args, data_path='datasets/'):

    with console.status("Reading data ...") as status:
        # get train and test dataset
        if args.dataset == 'mnist':
            tr_dataset, te_dataset = get_mnist(path=data_path+args.dataset)
        console.log(f"Finish fetching dataset: [green]{args.dataset}[/green]")
        dataset_size = len(tr_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.15 * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.bs, sampler=train_sampler, drop_last=True)
        va_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.bs, sampler=valid_sampler)
        te_loader = torch.utils.data.DataLoader(te_dataset, batch_size=args.bs)
        console.log(f"Finish generate dataloader")

        console.log(tr_dataset)
        data_dict = {}




def get_mnist(path:str):
    
    train_dataset = torchvision.datasets.MNIST(f'{path}/', train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize((0.1307,), (0.3081,))
                                                    ])
                                                )

    test_dataset = torchvision.datasets.MNIST(f'{path}/', train=False, download=True, 
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                    ])
                                                )
    
    return train_dataset, test_dataset