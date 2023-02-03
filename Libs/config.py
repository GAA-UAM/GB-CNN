import argparse

arg_list = []
parser = argparse.ArgumentParser()

def add_arg_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')