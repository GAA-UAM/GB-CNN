""" Model Configurations """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import argparse

arg_list = []
parser = argparse.ArgumentParser()


def add_arg_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


# general args
general_arg = add_arg_group("General")
general_arg.add_argument('--seed', type=int, default=111)

# CNN args
cnn_arg = add_arg_group("CNN")
cnn_arg.add_argument('--epoch', type=int, default=2)
cnn_arg.add_argument('--learning_rate', type=float, default=1e-3)
cnn_arg.add_argument('--batch_size', type=int, default=128)
cnn_arg.add_argument('--patience', type=int, default=3)


# GB args
gb_arg = add_arg_group("GB")
gb_arg.add_argument('--boosting_epoch', type=int, default=2)
gb_arg.add_argument('--additive_epoch', type=int, default=2)
gb_arg.add_argument('--batch_size_gb', type=int, default=128)
gb_arg.add_argument('--unit', type=int, default=1)
gb_arg.add_argument('--eta', type=float, default=1e-1)

def get_config():
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args
