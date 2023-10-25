import wandb
from typing import Dict

def init_tracker(name:str, config:Dict):
    wandb.init(project=name, 
               config=config)

def tracker_log(dct:Dict):
    wandb.log({**dct})

def tracker_log_table(dct:Dict, name:str):
    data = []
    for key in dct.keys():
        data.append([f'{key}', f'{dct[key]}'])
    my_table = wandb.Table(columns=["key", "value"], data=data)
    wandb.run.log({name: my_table})

def tracker_summary(dct:Dict):
    for key in dct.keys():
        wandb.run.summary[key] = dct[key]