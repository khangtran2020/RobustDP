import os
import sys
import torch
import datetime
from config import parse_args
from Data.read import read_data
from Utils.utils import print_args, seed_everything
from Utils.console import console
from Utils.tracking import init_tracker

def run(args, date, device):
    tr_loader, va_loader, te_loader = read_data(args=args)
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