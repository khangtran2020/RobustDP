import argparse


def add_general_group(group):
    group.add_argument("--model_path", type=str, default="results/models/", help="directory for saving model file")
    group.add_argument("--res_path", type=str, default="results/dict/", help="directory for output file")
    group.add_argument("--seed", type=int, default=2605, help="seed")
    group.add_argument("--general_mode", type=str, default='clean', help="Mode of running ['clean', 'dp']")
    group.add_argument("--device", type=str, default='cpu', help="device for running experiments")
    group.add_argument("--debug", type=int, default=0, help='running with debug mode or not')
    group.add_argument("--performance_metric", type=str, default='acc', help="Metrics of performance")
    group.add_argument("--project_name", type=str, default='', help="")


def add_data_group(group):
    group.add_argument('--data_path', type=str, default='Data/', help="dir path to dataset")
    group.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    group.add_argument("--data_mode", type=str, default='none', help="Mode for data processing")


def add_model_group(group):
    group.add_argument("--model_type", type=str, default='sage', help="Model type")
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument('--batch_size', type=int, default=512, help="batch size for training process")
    group.add_argument('--n_layers', type=int, default=2, help='# of layers')
    group.add_argument('--hid_dim', type=int, default=64, help='hidden embedding dim')
    group.add_argument("--optimizer", type=str, default='adam')
    group.add_argument("--dropout", type=float, default=0.2)
    group.add_argument("--patience", type=int, default=20)
    group.add_argument("--epochs", type=int, default=100, help='training step')

def add_dp_group(group):
    group.add_argument("--ns", type=float, default=1.0, help='noise scale for dp')
    group.add_argument("--clip", type=float, default=1.0, help='clipping gradient bound')
    group.add_argument('--sampling_rate', type=float, default=0.08, help="batch size for training process")

def parse_args():
    parser = argparse.ArgumentParser()
    exp_grp = parser.add_argument_group(title="Attack setting")

    add_general_group(exp_grp)
    add_data_group(exp_grp)
    add_model_group(exp_grp)
    add_dp_group(exp_grp)
    return parser.parse_args()