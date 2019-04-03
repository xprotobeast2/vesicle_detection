import os
import sys
import numpy as np
import json
import torch
from torch import nn

def make_checkpoint(model, optimizer, epoch):
    chk = {}
    if isinstance(model, nn.DataParallel):
        chk['model'] = model.module.state_dict()
    else:
        chk['model'] = model.state_dict()
    chk['optimizer'] = optimizer.state_dict()
    chk['epoch'] = epoch
    return chk


def periodic_integer_delta(inp, every=10, start=-1):
    return (inp % every) == ((start + every) % every)


def dump_everything(args):
    # Destination:
    destination_dir = os.path.join(args.base_output)
    destination_file = os.path.join(destination_dir, "args.json")
    obj = {k: v for k, v in args.__dict__.items()}
    with open(destination_file, 'w') as f:
        json.dump(obj, f, indent="\t")
