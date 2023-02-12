""" GB-CNN Configurations """

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
gb_arg.add_argument('--boosting_epoch', type=int, default=40)
gb_arg.add_argument('--boosting_eta', type=float, default=1e-1)
# Note, save_records will consume more memory
gb_arg.add_argument('--save_records', type=str2bool, default=False,
                    help='save trained models, and additional training metrics')


# fine_tune args
ft_arg = add_arg_group("additive_model")
ft_arg.add_argument('--additive_epoch', type=int, default=200)
ft_arg.add_argument('--batch', type=int, default=128)
ft_arg.add_argument('--units', type=int, default=20)
ft_arg.add_argument('--additive_eta', type=float, default=1e-3)
ft_arg.add_argument('--patience', type=int, default=3)


def get_config():
    args, _ = parser.parse_known_args()
    return args
