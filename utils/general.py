import os
import sys
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
    destination_dir = os.path.join(args.base_output, args.run_code)
    destination_file = os.path.join(destination_dir, "information.json")
    obj = {}
    args_serializable = {k: v for k, v in args.__dict__.items() if ((k != "meta") and (k != "wtree"))}
    args_serializable["meta"] = {k: v for k, v in args.__dict__["meta"].items() if ((k != "s2mu") and (k != "s2std"))}
    args_serializable["meta"]["s2mu"] = {k: v.tolist() for k, v in args.__dict__["meta"]["s2mu"].items()}
    args_serializable["meta"]["s2std"] = {k: v.tolist() for k, v in args.__dict__["meta"]["s2std"].items()}
    obj["args"] = args_serializable
    with open(destination_file, 'w') as f:
        json.dump(obj, f, indent="\t")
