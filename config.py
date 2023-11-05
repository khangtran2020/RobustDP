import argparse


def add_general_group(group):
    group.add_argument("--model_path", type=str, default="results/models/", help="directory for saving model file")
    group.add_argument("--res_path", type=str, default="results/dict/", help="directory for output file")
    group.add_argument("--seed", type=int, default=2605, help="seed")
    group.add_argument("--gen_mode", type=str, default='clean', help="Mode of running ['clean', 'dp']")
    group.add_argument("--device", type=str, default='cpu', help="device for running experiments")
    group.add_argument("--debug", type=int, default=0, help='running with debug mode or not')
    group.add_argument("--metric", type=str, default='acc', help="Metrics of performance")
    group.add_argument("--proj_name", type=str, default='', help="", required=True)

def add_data_group(group):
    group.add_argument('--data_path', type=str, default='Data/dataset/', help="dir path to dataset")
    group.add_argument('--data', type=str, default='mnist', help="name of dataset")
    group.add_argument("--data_mode", type=str, default='none', help="Mode for data processing")
    group.add_argument('--img_sz', type=int, default=28, help='Size of a square image')

def add_model_group(group):
    group.add_argument("--model", type=str, default='cnn', help="Model type")
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument('--bs', type=int, default=512, help="batch size for training process")
    group.add_argument('--nlay', type=int, default=2, help='# of layers')
    group.add_argument('--hdim', type=int, default=64, help='hidden embedding dim')
    group.add_argument("--opt", type=str, default='adam')
    group.add_argument("--dout", type=float, default=0.2)
    group.add_argument("--pat", type=int, default=20)
    group.add_argument("--clipw", type=float, default=1.0, help='clipping bound for lipschitz condtion')
    group.add_argument("--epochs", type=int, default=100, help='training step')

def add_dp_group(group):
    group.add_argument("--eps", type=float, default=1.0, help='target privacy budget')
    group.add_argument("--clip", type=float, default=1.0, help='clipping gradient bound')
    group.add_argument('--max_bs', type=int, default=128, help="max physical batch size for opacus / dpsgd training")
    group.add_argument('--num_mo', type=int, default=100, help="# output model")

def add_model_attack_group(group):
    group.add_argument("--att_mode", type=str, default='fsgm-clean', help="Attack mode", required=True)
    group.add_argument("--pgd_steps", type=int, default=50, help='training step for pgd')

def parse_args():
    parser = argparse.ArgumentParser()
    exp_grp = parser.add_argument_group(title="Attack setting")

    add_general_group(exp_grp)
    add_data_group(exp_grp)
    add_model_group(exp_grp)
    add_dp_group(exp_grp)
    add_model_attack_group(exp_grp)
    return parser.parse_args()