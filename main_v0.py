'''
Quick baseline
for: Image segmentation (into vesicles)
on: Sliding windows on original images
using: UNet
'''


import os
import sys
import time
import pickle
import multiprocessing
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision import transforms
from tqdm import tqdm
from utils import (
    dataset as custom_dataset,
    general as general_utils,
    utils as seg_utils
)
from modules import unet, loss_fns
import sklearn.metrics


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_path', help='Path to dataset of ordered vertices and adjacencies')
    parser.add_argument('--model_path', help='Path to model weights')
    parser.add_argument('-t', '--test_frac', default=0.3, help='Fraction of data to use for testing')
    parser.add_argument('-d', '--debug', action='store_true', help='If set, then use debugging mode - dataset consists of a small number of points')
    parser.add_argument('-p', '--patch_size', type=int, default=128, help='Input data is split into windows of size patch_size')  
    parser.add_argument('-s', '--patch_step', type=int, default=64, help='Input data step size') 
    parser.add_argument('-a', '--augment', action='store_true', help='Apply data augmentation') 
    
    # Output
    parser.add_argument("--base_output", dest="base_output", default="./output/seg", help="Directory which will have folders per run")  # noqa
    parser.add_argument("-r", "--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
    parser.add_argument('--eval_every', dest='eval_every', default=1, help='How often to evaluate model')
    parser.add_argument('--save_every', dest='save_every', default=1, help='How often to save model')

    # Hyperparameters
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa
    parser.add_argument('-e', '--epochs', type=int, default=12, help='Number of epochs for training')
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>', default=64, help="Batch size (default=64)")  # noqa
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, metavar='<float>', default=3e-4, help='Learning rate')  # noqa
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0, help='Weight decay')  # noqa
    parser.add_argument("-ud", "--unet_depth", dest="unet_depth", type=int, metavar='<int>', default=2, help='Number of unet encoding/decoding blocks')  # noqa
    parser.add_argument("-nc", "--num_classes", dest="num_classes", type=int, metavar='<int>', default=2, help='Number of classes for segmentation')  # noqa
    parser.add_argument("-fs", "--feat_scale", dest="feat_scale", type=int, metavar='<int>', default=1, help='Feature map scaling factor for each level')  # noqa

    # Optimization
    parser.add_argument("--loss_func", choices=['dice', 'crossentropy'], default='dice', help='Choose the loss function')  # noqa
    parser.add_argument("-cw", "--class_weights", action='store_true', help='Whether the loss should be weighted by class proportion' )
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
    os.makedirs(args.base_output+'/losses', exist_ok=True)
    print("Using run_code: {}".format(args.run_code))
    return args


replace_metric_by_mean = ['loss', 'dice']

def load_model(args, default_model=unet.UNetAuto):
    if args.model_path is None:
        return default_model(n_classes=args.num_classes, depth=args.unet_depth, feat_scale=args.feat_scale)
    model_dict = torch.load(args.model_path)
    model = default_model(n_classes=args.num_classes, depth=args.unet_depth, feat_scale=args.feat_scale)
    model.load_state_dict(model_dict["model"])
    return model

def map_segmentation(seg, n_classes=2):
    # Depending on how many classes are being used, we modify the segmentation differently
    if n_classes > 2:
        m1 = seg < 8
        m2 = seg > 0
        seg[m1 & m2] = 1
        seg[seg == 8] = 2
    elif n_classes == 2:
        seg = (seg > 0)
    return seg

def dice_score(preds, targets):

    intersection = torch.einsum('bcij, bcij -> bc', [preds, targets])
    union = torch.einsum('bcij, bcij -> bc', [preds, preds]) + \
            torch.einsum('bcij, bcij -> bc', [targets, targets]) + 1e-16

    iou = torch.div(intersection, union)
    dice = 2*iou
    
    avg_dice = (torch.einsum('bc->', dice) / (targets.shape[0]*targets.shape[1]))
    if torch.isnan(iou).sum():
        print(iou)
        print(torch.isnan(intersection).sum(),torch.isnan(union).sum())
    return avg_dice

def get_metrics(seg, seg_hat):
    metrics = {}
    with torch.no_grad():
        seg = seg.cpu()
        target = seg_utils.argmax_to_categorical(seg_hat).cpu()
        #seg_hat = seg_hat.view(-1).cpu().numpy()
        # metrics['cm'] = np.array(sklearn.metrics.confusion_matrix(seg, seg_hat))
        # metrics['accuracy'] = np.array(sklearn.metrics.accuracy_score(seg, target))
        # metrics['precision'] = np.array(sklearn.metrics.precision_score(seg, target))
        # metrics['recall'] = np.array(sklearn.metrics.recall_score(seg, target))
        metrics['dice'] = dice_score(seg, target)
        # metrics['kappa'] = sklearn.metrics.cohen_kappa_score(seg, seg_hat)
    return metrics


def test(args, model, loader, prefix='', verbose=True):
    
    print("train: Beginning test")

    metrics = defaultdict(list)
    t = tqdm(loader)
    with torch.no_grad():
        for (img, seg) in t:
            img = img.to(args.device)
            seg = map_segmentation(seg, args.num_classes).long()          
            seg_hot = seg_utils.one_hot_encode(seg, args.num_classes).to(args.device)    
            seg = seg.to(args.device)

            seg_hat = model(img)
            
            if args.loss_func=='dice':
                dice_loss = loss_fns.DiceLoss()
                loss = dice_loss(seg_hat, seg_hot)
            elif args.loss_func=='crossentropy':
                log_loss = nn.CrossEntropyLoss()
                loss = log_loss(seg_hat, seg)

            t.set_postfix_str(s='loss: %f'% loss.item())

            try:    
                metrics['loss'].append(loss.item())
            except ValueError:
                print(metrics)
            for k, v in get_metrics(seg_hot, seg_hat).items():
                metrics[k].append(v)

        for k in replace_metric_by_mean:
            metrics[k] = np.mean(metrics[k])

        # Print!
        if verbose:
            start_string = '#### {} evaluation ####'.format(prefix)
            print(start_string)
            for k, v in metrics.items():
                print('#### {} = {}'.format(k, v))
            print(''.join(['#' for _ in range(len(start_string))]))
    return metrics


def train_unet_v0(args, dataset, train_loader, test_loader):
    output_dir = args.base_output
    model = load_model(args, unet.FRUNetAuto)
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

    print("Getting Class balances... ")

    class_weights = None
    if args.class_weights:
        class_weights = [dataset.class_weights[0], dataset.class_weights[1:8].sum()]
        if args.num_classes > 2:
            class_weights.append(dataset.class_weights[-1])
        class_weights = 1 - torch.Tensor(class_weights)
        class_weights = class_weights.to(args.device)

    print("Class balance: %s"%str(class_weights))
    print("train: Beginning train")

    model.train()

    for epoch_idx in range(args.epochs):
        print('Starting epoch {}'.format(epoch_idx))
        epoch_metrics = defaultdict(list)
        tic = time.time()
        t = tqdm(train_loader)
        for (img, seg) in t:
            img = img.to(args.device)
            seg = map_segmentation(seg, args.num_classes).long()
            seg_hot = seg_utils.one_hot_encode(seg, args.num_classes).to(args.device)
            seg = seg.to(args.device)
            
            seg_hat = model(img)

            if args.loss_func=='dice':
                dice_loss = loss_fns.DiceLoss(class_weights)
                loss = dice_loss(seg_hat, seg_hot)
            elif args.loss_func=='crossentropy':
                log_loss = nn.CrossEntropyLoss(weight=class_weights)
                loss = log_loss(seg_hat, seg)

            epoch_metrics['loss'].append(loss.item())
            for k, v in get_metrics(seg_hot, seg_hat).items():
                epoch_metrics[k].append(v)

            t.set_postfix_str(s='loss: %f, dice: %f'%(loss.item(),epoch_metrics['dice'][-1]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for k, v in epoch_metrics.items():
            metrics[k].append(np.mean(np.array(v), axis=0))
            epoch_metrics[k].append(np.mean(np.array(v), axis=0))

        print('#### [{:.2f}] Epoch {} ####'.format(time.time() - tic, epoch_idx))
        for k, v in epoch_metrics.items():
            print('#### {}: {}'.format(k, v[-1]))
        print('#####################')

        # Eval and save if necessary.
        if general_utils.periodic_integer_delta(epoch_idx, args.eval_every):
            model = model.eval()
            test_metrics = test(args, model, test_loader, prefix='Test Dataset, Epoch {}'.format(epoch_idx))
            model = model.train()
            for k, v in test_metrics.items():
                metrics['test_{}_epoch{}'.format(k, epoch_idx)] = v

        if general_utils.periodic_integer_delta(epoch_idx, args.save_every):
            checkpoint_path = os.path.join(output_dir, "last.checkpoint")
            print('Saving model to {}'.format(checkpoint_path))
            chk = general_utils.make_checkpoint(model, optimizer, epoch_idx)
            torch.save(chk, checkpoint_path)
            np.save(output_dir+'/losses/loss%d.npy'%epoch_idx, epoch_metrics['loss'])
    return model, metrics


def run_v0(args):
    
    print("main: Creating dataset")

    seg_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])

    dataset = custom_dataset.PixelSegmentationDataset(args.data_path, 
        cpu_count=args.num_workers, window_size=args.patch_size, step_size=args.patch_step,
        n_classes=args.num_classes, transform=(seg_transforms if args.augment else None))
    if args.debug:
        dataset.n = 50

    indices = np.random.permutation(list(range(len(dataset))))
    split_pos = int(args.test_frac * len(dataset))

    print("main: Creating train loader")

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=torch.utils.data.SubsetRandomSampler(indices[split_pos:]),
    )
    
    print("main: Creating test loader")

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
    with open(os.path.join(args.base_output, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
