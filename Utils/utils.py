import os
import torch
import time
import pickle
import random
import numpy as np
from contextlib import contextmanager
from Utils.console import console, log_table

@contextmanager
def timeit(logger, task):
    logger.info(f'Started task {task} ...')
    t0 = time.time()
    yield
    t1 = time.time()
    logger.info(f'Completed task {task} - {(t1 - t0):.3f} sec.')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_name(args, current_date):

    date_str = f'{current_date.day}{current_date.month}{current_date.year}-{current_date.hour}{current_date.minute}'
    data_keys = ['data', 'seed', 'data_mode']
    model_keys = ['data', 'gen_mode', 'seed', 'model', 'lr', 'nlay', 'hdim', 'epochs', 'opt']
    gen_keys = ['data', 'gen_mode', 'data_mode', 'seed', 'model', 'nlay', 'hdim']

    general_str = ''
    for key in gen_keys:
        general_str += f"{key}_{getattr(args, key)}_"
    general_str += date_str

    data_str = ''
    for key in data_keys:
        data_str += f"{key}_{getattr(args, key)}_"
    
    model_str = ''
    for key in model_keys:
        model_str += f"{key}_{getattr(args, key)}_"

    name = {
        'data': data_str[:-1],
        'model': model_str[:-1],
        'general': general_str
    }

    return name

def save_dict(path, dct):
    with open(path, 'wb') as f:
        pickle.dump(dct, f)

def get_index_by_value(a, val):
    return (a == val).nonzero(as_tuple=True)[0]

def get_index_bynot_value(a, val):
    return (a != val).nonzero(as_tuple=True)[0]

def get_index_by_list(arr, test_arr):
    return torch.isin(arr, test_arr).nonzero(as_tuple=True)[0]

def get_index_by_not_list(arr, test_arr):
    return (1 - torch.isin(arr, test_arr).int()).nonzero(as_tuple=True)[0]

def print_args(args):
    arg_dict = {}
    keys = ['gen_mode', 'data', 'data_mode', 'proj_name', 'img_sz', 'bs', 'debug', 'model', 'lr', 'bs', 
            'nlay', 'hdim', 'opt', 'dout', 'epochs']
    for key in keys:
        arg_dict[f'{key}'] = f'{getattr(args, key)}'
    log_table(dct=arg_dict, name='Arguments')
    return arg_dict

def read_pickel(file):
    with open(file, 'rb') as f:
        res = pickle.load(f)
    return res

def init_history(args):

    data_hist = {
        'tr_id': None,
        'va_id': None,
        'te_id': None,
    }

    target_model_hist = {
        'name': None,
        'tr_loss': [],
        'tr_perf': [],
        'va_loss': [],
        'va_perf': [],
        'te_loss': [],
        'te_perf': [],
        'best_test': 0
    }
    
    att_hist = {
        'att_loss': [],
        'att_asr': [],
        'conv_perf': [],
        'cert_perf': []
    }

    return data_hist, target_model_hist, att_hist