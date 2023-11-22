import os
import sys
import torch
import datetime
import warnings
from config import parse_args
from Data.read import read_data
from Models.utils import init_model
from Runs.clean import train, evalt
from Runs.dpsgd import traindp, evaltdp
from Attacks.utils import robust_eval_clean, robust_eval_dp
from Utils.utils import print_args, seed_everything, init_history, get_name, save_dict
from Utils.console import console
from Utils.tracking import init_tracker

warnings.filterwarnings("ignore")

def run(args, date, device):
    
    data_hist, model_hist, att_hist = init_history(args=args)
    name = get_name(args=args, current_date=date)
    tr_loader, va_loader, te_loader = read_data(args=args)
    model = init_model(args=args)
    
    if args.debug > 0:
        for n, p in model.named_children():
            console.log(f"Layer {n}: {p}")

        console.log(model.state_dict)
        with console.status("Testing comparable of considered model") as status:
            with torch.no_grad():
                console.log(f"Model: {model}")
                for n, p in model.named_parameters():
                    console.log(f"Size of param {n}: {p.size()}") 
                image, _ = next(iter(tr_loader))
                out_put = model(image)
                console.log(f"Output dimension: {out_put.size()}, with value: {out_put}")
                console.log(f'[bold][green]Done testing comparable of considered mode: :white_check_mark:')
        # model = lip_clip(model=model, clip=args.clipw)
        # model = clip_weight(model=model, clip=args.clipw)
        # checked = check_clipped(model=model, clip=args.clipw)
    
    # train the model
    if args.gen_mode == 'clean':
        model, model_hist = train(args=args, tr_loader=tr_loader, va_loader=va_loader, model=model, device=device, history=model_hist, name=name['model'])
        model_hist = evalt(args=args, te_loader=te_loader, model=model, device=device, history=model_hist)
        torch.cuda.empty_cache()
        robust_eval_clean(args=args, model=model, device=device, te_loader=te_loader, num_plot=50, history=att_hist)
    else:
        model_list, model_hist = traindp(args=args, tr_loader=tr_loader, va_loader=va_loader, model=model, device=device, history=model_hist, name=name['model'])
        model_hist = evaltdp(args=args, te_loader=te_loader, model_list=model_list, device=device, history=model_hist)
        torch.cuda.empty_cache()
        robust_eval_dp(args=args, model_list=model_list, device=torch.device('cpu'), te_loader=te_loader, num_plot=50, history=att_hist)

    general_hist = {
        'data': data_hist,
        'model': model_hist,
        'att': att_hist
    }
    general_path = args.res_path + f"{name['general']}.pkl"
    save_dict(path=general_path, dct=general_hist)
    console.log(f"Saved result at path {general_path}.")

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