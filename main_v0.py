'''
Quick baseline
for: Image segmentation (into vesicles)
on: Sliding windows on original images
using: UNet
'''


import os
import sys
import time
import multiprocessing
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from utils import (
    dataset as custom_dataset,
    general as general_utils,
)
from modules import unet
import sklearn.metrics


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('data_path', help='Path to dataset of ordered vertices and adjacencies')
    parser.add_argument('-t', '--test_frac', default=0.3, help='Fraction of data to use for testing')
    parser.add_argument('-d', '--debug', action='store_true', help='If set, then use debugging mode - dataset consists of a small number of points')

    # Output
    parser.add_argument("--base_output", dest="base_output", default="./output/", help="Directory which will have folders per run")  # noqa
    parser.add_argument("-r", "--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
    parser.add_argument('--eval_every', dest='eval_every', default=1, help='How often to evaluate model')
    parser.add_argument('--save_every', dest='save_every', default=1, help='How often to save model')

    # Hyperparameters
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")  # noqa
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, metavar='<float>', default=0.001, help='Learning rate')  # noqa
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0, help='Weight decay')  # noqa

    # Hardware
    parser.add_argument('--cuda', action='store_true', help='Use GPU if available or not')
    parser.add_argument("-dp", "--dataparallel", dest="dataparallel", default=False, action="store_true")  # noqa

    args = parser.parse_args()

    args.num_workers = multiprocessing.cpu_count() // 3
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    args.dataparallel = args.dataparallel and args.cuda
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.debug:
        args.run_code = "debug"
    os.makedirs(args.base_output, exist_ok=True)
    if len(args.run_code) == 0:
        # Generate a run code by counting number of directories in oututs
        run_count = len(os.listdir(args.base_output))
        args.run_code = 'run{}'.format(run_count)
    args.base_output = os.path.join(args.base_output, args.run_code)
    os.makedirs(args.base_output, exist_ok=True)
    print("Using run_code: {}".format(args.run_code))
    return args


replace_metric_by_mean = ['loss', 'cm', 'accuracy', 'precision', 'recall', 'f1']


def get_metrics(seg, seg_hat):
    metrics = {}
    with torch.no_grad():
        seg = seg.view(-1).cpu().numpy()
        seg_hat = torch.argmax(seg_hat, dim=1).view(-1).cpu().numpy()
        # metrics['cm'] = np.array(sklearn.metrics.confusion_matrix(seg, seg_hat))
        metrics['accuracy'] = np.array(sklearn.metrics.accuracy_score(seg, seg_hat))
        metrics['precision'] = np.array(sklearn.metrics.precision_score(seg, seg_hat))
        metrics['recall'] = np.array(sklearn.metrics.recall_score(seg, seg_hat))
        metrics['f1'] = np.array(sklearn.metrics.f1_score(seg, seg_hat))
        # metrics['kappa'] = sklearn.metrics.cohen_kappa_score(seg, seg_hat)
    return metrics


def test(args, model, loader, prefix='', verbose=True):
    metrics = defaultdict(list)
    with torch.no_grad():
        for bidx, (img, seg) in enumerate(loader):
            img = img.to(args.device)
            seg = (seg.to(args.device) > 0).long()
            seg_hat = model(img)
            loss = F.cross_entropy(seg_hat, seg)
            metrics['loss'].append(loss.item())
            for k, v in get_metrics(seg, seg_hat).items():
                metrics[k].append(v)

        for k in replace_metric_by_mean:
            metrics[k] = np.mean(metrics[k])

        # Print!
        if verbose:
            start_string = '#### {} evaluation ####'.format(prefix)
            print(start_string)
            for k, v in metrics.items():
                print('#### {} = {:.3f}'.format(k, v))
            print(''.join(['#' for _ in range(len(start_string))]))
    return metrics


def train_unet_v0(args, dataset, train_loader, test_loader):
    output_dir = os.path.join(args.base_output, args.run_code)
    model = unet.UNet128()
    if args.cuda:
        model = model.cuda()
    if args.dataparallel:
        model = nn.DataParallel(model)

    params = list(model.parameters())
    optimizer = optim.Adam(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    metrics = defaultdict(list)
    for epoch_idx in range(args.epochs):
        print('Starting epoch {}'.format(epoch_idx))
        epoch_metrics = defaultdict(list)
        tic = time.time()
        for bidx, (img, seg) in enumerate(train_loader):
            img = img.to(args.device)
            seg = (seg.to(args.device) > 0).long()
            seg_hat = model(img)

            loss = F.cross_entropy(seg_hat, seg)

            epoch_metrics['loss'].append(loss.item())
            for k, v in get_metrics(seg, seg_hat).items():
                epoch_metrics[k].append(v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for k, v in epoch_metrics.items():
            metrics[k].append(np.mean(v))

        print('#### [{:.2f}] Epoch {} ####'.format(time.time() - tic, epoch_idx))
        for k, v in metrics.items():
            print('#### {}: {:.3f}'.format(k, v[epoch_idx]))
        print('#####################')

        # Eval and save if necessary.
        if general_utils.periodic_integer_delta(epoch_idx, args.eval_every):
            test_metrics = test(args, model, test_loader, prefix='Test Dataset, Epoch {}'.format(epoch_idx))
            for k, v in test_metrics.items():
                metrics['test_{}_epoch{}'.format(k, epoch_idx)] = v

        if general_utils.periodic_integer_delta(epoch_idx, args.save_every):
            checkpoint_path = os.path.join(output_dir, "last.checkpoint")
            print('Saving model to {}'.format(checkpoint_path))
            chk = general_utils.make_checkpoint(model, optimizer, epoch)
            torch.save(chk, checkpoint_path)
    return model, metrics


def run_v0(args):
    dataset = custom_dataset.PixelSegmentationDataset(args.data_path, load_from_npz=None, cpu_count=args.num_workers)
    if args.debug:
        dataset.n = 50

    indices = np.random.permutation(list(range(len(dataset))))
    split_pos = int(args.test_frac * len(dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=torch.utils.data.SubsetRandomSampler(indices[split_pos:]),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=torch.utils.data.SubsetRandomSampler(indices[:split_pos]),
    )
    model, metrics = train_unet_v0(args, dataset, train_loader, test_loader)
    return model, metrics


if __name__ == '__main__':
    args = get_args()
    general_utils.dump_everything(args)
    model, metrics = run_v0(args)
    with open(os.path.join(args.base_output, args.run_code, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
