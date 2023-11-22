import torch
import torchvision
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from Utils.console import console, log_table
from opacus.data_loader import DPDataLoader
from Data.dataset import Data

def read_data(args, data_path='datasets/'):

    with console.status("Reading data ...") as status:
        # get train and test dataset

        if args.data == 'mnist':
            tr_dataset, te_dataset = get_mnist(path=data_path+args.data, size=args.img_sz)
            target = tr_dataset.targets
        elif args.data == 'cifar10':
            tr_dataset, te_dataset = get_cifar(path=data_path+args.data, size=args.img_sz)
            target = torch.Tensor(tr_dataset.targets)
        elif args.data == 'utk':
            tr_dataset, te_dataset = get_utk(seed=args.seed)
            target = tr_dataset.y.int().clone()

        args.num_class = target.unique().size(dim=0)
        target = target.tolist()
        console.log(f"Finish fetching dataset: [green]{args.data}[/green]")
        dataset_size = len(tr_dataset)
        indices = list(range(dataset_size))
        id_tr, id_va, _, _ = train_test_split(indices, target, test_size=0.15, stratify=target)
        train_sampler = SubsetRandomSampler(id_tr)
        valid_sampler = SubsetRandomSampler(id_va)

        
        va_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.trbs, sampler=valid_sampler)
        te_loader = torch.utils.data.DataLoader(te_dataset, batch_size=args.tebs)
        if args.gen_mode == 'dp':
            tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=int(args.sp_rate * len(id_tr)), sampler=train_sampler, shuffle=True, drop_last=True)
            # tr_loader = DPDataLoader.from_data_loader(tr_loader, generator=None, distributed=False)
        else:
            tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.trbs, sampler=train_sampler, shuffle=True, drop_last=True)
        console.log(f"Finish generate dataloader")

        image, _ = next(iter(tr_loader))
        args.channel_in = image.size()[1]
        data_dict = {
            '# data train': f"{len(id_tr)}",
            '# data valid': f"{len(id_va)}",
            '# data test': f"{torch.Tensor(te_dataset.y.int().clone()).size(dim=0) if args.data == 'utk' else torch.Tensor(te_dataset.targets).size(dim=0)}",
            '# features':  f"{image.size()[1:]}",
            '# labels': f"{args.num_class}",
            'batch size': f"{image.size(dim=0)}"
        }
        log_table(dct=data_dict, name=f"{args.data}'s Property")
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

def get_cifar(path:str, size:int):

    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
        transforms.Resize(size=size)
    ])

    tr_dataset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)

    te_dataset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

    return tr_dataset, te_dataset

def get_utk(seed:int):

    utk_data_path = "datasets/utk/age_gender.gz"
    label = 'gender'
    protect = 'gender'
    pd00 = pd.read_csv(utk_data_path, compression='gzip')
    pd00[label] = pd00[label].apply(lambda x: x!=0).astype(int)
    n_class = 2
    y = pd00[label].values
    np.random.seed(0)  # random seed of partition data into train/test
    tr_df, te_df, _, _  = train_test_split(pd00, y,  test_size=0.2, stratify=y, random_state=seed)
    tr_df = tr_df.reset_index(drop=True)
    te_df = te_df.reset_index(drop=True)

    X_tr = get_pixel(df=tr_df)
    X_te = get_pixel(df=te_df)

    tr_dataset = Data(X=X_tr, y=tr_df[label].values)
    te_dataset = Data(X=X_te, y=te_df[label].values)

    return tr_dataset, te_dataset

def get_pixel(df:pd.DataFrame):
    X = df.pixels.apply(lambda x: np.array(x.split(" "), dtype=float))
    X = np.stack(X)
    X = X / 255.0
    X = X.astype('float32').reshape(X.shape[0], 1, 48, 48)
    return X

