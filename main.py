import os
import sys
import torch
import datetime
from config import parse_args
from Data.read import read_data
from Models.model import CNN
from Utils.utils import print_args, seed_everything
from Utils.console import console
from Utils.tracking import init_tracker

def run(args, date, device):
    
    tr_loader, va_loader, te_loader = read_data(args=args)
    model = CNN(channel=[32, 32, 64], hid_dim=[256], img_size=args.img_size, channel_in=args.channel_in, out_dim=args.num_label, kernal_size=5)
    console.log(f"Model: {model}")

    if args.debug > 0:
        with console.status("Testing comparable of considered model") as status:
            image, _ = next(iter(tr_loader))
            out_put = model(image)
            console.log(f"Output dimension: {out_put.size()}")
            console.log(f'[bold][green]Done!')


    # sys.exit()

if __name__ == "__main__":
    date = datetime.datetime.now()
    args = parse_args()
    console.rule(f"Begin experiment: {args.project_name}")
    with console.status("Initializing...") as status:
        args_dict = print_args(args=args)
        args.debug = True if args.debug == 1 else False
        seed_everything(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.device == 'cpu':
            device = torch.device('cpu')
        console.log(f"DEVICE USING: {device}")
        init_tracker(name=args.project_name, config=args_dict)
        console.log(f'[bold][green]Done!')

    run(args=args, date=date, device=device)
    console.rule(f"Finished experiment: {args.project_name}")