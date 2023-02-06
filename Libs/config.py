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

# GB args
gb_arg = add_arg_group("GB")
gb_arg.add_argument('--boosting_epoch', type=int, default=200)
gb_arg.add_argument('--boosting_eta', type=float, default=1e-1)
gb_arg.add_argument('--boosting_patience', type=int, default=3)

# fine_tune args
ft_arg = add_arg_group("additive_model")
ft_arg.add_argument('--additive_epoch', type=int, default=100)
ft_arg.add_argument('--additive_batch', type=int, default=64)
ft_arg.add_argument('--additive_units', type=int, default=10)
ft_arg.add_argument('--additive_eta', type=float, default=1e-3)
ft_arg.add_argument('--additive_patience', type=int, default=7)


def get_config():
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args
