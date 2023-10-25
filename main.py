import os
import sys
import torch
import datetime
from config import parse_args
from Data.read import read_data
from Models.model import CNN
from Runs.clean import train, evalt
from Utils.utils import print_args, seed_everything, init_history, get_name
from Utils.console import console
from Utils.tracking import init_tracker

def run(args, date, device):
    
    data_hist, model_hist, att_hist = init_history(args=args)
    name = get_name(args=args, current_date=date)
    tr_loader, va_loader, te_loader = read_data(args=args)
    model = CNN(channel=[32, 32, 64], hid_dim=[256], img_size=args.img_size, channel_in=args.channel_in, out_dim=args.num_class, kernal_size=5)

    if args.debug > 0:
        with console.status("Testing comparable of considered model") as status:
            console.log(f"Model: {model}")
            for n, p in model.named_parameters():
                console.log(f"Size of param {n}: {p.size()}") 
            image, _ = next(iter(tr_loader))
            out_put = model(image)
            console.log(f"Output dimension: {out_put.size()}")
            console.log(f'[bold][green]Done testing comparable of considered mode: :white_check_mark:')
    
    # train the model
    if args.gen_mode == 'clean':
        model, model_hist = train(args=args, tr_loader=tr_loader, va_loader=va_loader, model=model, device=device, history=model_hist, name=name['model'])



    # sys.exit()

if __name__ == "__main__":
    date = datetime.datetime.now()
    args = parse_args()
    console.rule(f"Begin experiment: {args.proj_name}")
    with console.status("Initializing...") as status:
        args_dict = print_args(args=args)
        args.debug = True if args.debug == 1 else False
        seed_everything(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.device == 'cpu':
            device = torch.device('cpu')
        console.log(f"DEVICE USING: {device}")
        init_tracker(name=args.proj_name, config=args_dict)
        console.log(f'[bold][green]Done Initializing!')

    run(args=args, date=date, device=device)
    console.rule(f"Finished experiment: {args.proj_name}")