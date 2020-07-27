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
from tqdm import tqdm
from utils import (
    dataset as custom_dataset,
    general as general_utils,
)
from modules import (
    detector,
    loss_fns,
)
import sklearn.metrics
from utils.bbox_utils import bbox_iou_numpy, compute_ap


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_path', help='Path to dataset of ordered vertices and adjacencies')
    parser.add_argument('-ld', '--load_data', action='store_true', help='If set, then create dataset from image list')
    parser.add_argument('-t', '--test_frac', default=0.3, help='Fraction of data to use for testing')
    parser.add_argument('-d', '--debug', action='store_true', help='If set, then use debugging mode - dataset consists of a small number of points')

    # Output
    parser.add_argument("--base_output", dest="base_output", default="./output/detector", help="Directory which will have folders per run")  # noqa
    parser.add_argument("-r", "--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
    parser.add_argument('--eval_every', dest='eval_every', default=1, help='How often to evaluate model')
    parser.add_argument('--save_every', dest='save_every', default=1, help='How often to save model')

    # Pretrained model.
    parser.add_argument('--config_path', type=str, default='yolov3/config/yolov3.cfg', help='Path to YOLO config file')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to YOLO weights file')

    # Hyperparameters
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>', default=8, help="Batch size (default=32)")  # noqa
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, metavar='<float>', default=0.00025, help='Learning rate')  # noqa
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0, help='Weight decay')  # noqa

    # Hardware
    parser.add_argument('--cuda', action='store_true', help='Use GPU if available or not')
    parser.add_argument("-dp", "--dataparallel", dest="dataparallel", default=False, action="store_true")  # noqa

    args = parser.parse_args()

    args.num_workers = multiprocessing.cpu_count() // 4
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


replace_metric_by_mean = ['loss', 'AP[1]', 'mAP']


def load_model(args, default_model=detector.Detector):
    if args.model_path is None:
        return default_model(
                args.config_path,
                dataset.window_size,
                weights_path=args.weights_path,
                n_classes=dataset.n_classes)
    model_dict = torch.load(args.model_path)
    model = default_model(
                args.config_path,
                dataset.window_size,
                weights_path=args.weights_path,
                n_classes=dataset.n_classes)
    model.load_state_dict(model_dict["model"])
    return model

def get_metrics(det, det_hat, min_label=1, iou_thres=0.4):
    pred_boxes, pred_cls = det_hat
    metrics = {}
    pred_boxes, pred_cls = det_hat
    n_classes = pred_cls.shape[-1]
    batch_size = det.shape[0]

    preds_per_idx = []
    truth_per_idx = []
    with torch.no_grad():
        for b in range(batch_size):
            preds_per_label = [np.array([]) for i in range(n_classes)]
            truth_per_label = [np.array([]) for i in range(n_classes)]

            # Figure out predictions
            np_pred_box = pred_boxes[b].cpu().numpy()
            np_scores = np_pred_box[:, 4]
            sort_i = np.argsort(np_scores)
            np_pred_box = np_pred_box[sort_i]
            np_pred_cls = np.argmax(pred_cls[b].cpu().numpy()[sort_i], axis=1)

            # Transform
            # pred is in x1, y1, w, h
            np_pred_box[:, 2] += np_pred_box[:, 0]
            np_pred_box[:, 3] += np_pred_box[:, 1]

            for label in range(n_classes):
                preds_per_label[label] = np_pred_box[np_pred_cls == label]

            # Figure out truths
            np_truth_box = det[b, :, 1:].cpu().numpy()
            mask = np_truth_box.sum(1) > 0
            np_truth_box = np_truth_box[mask]
            np_truth_cls = det[b, :, 0].cpu().numpy()[mask]

            # Transform
            # truth_box is in x1, y1, x2, y2 format.

            for label in range(n_classes):
                truth_per_label[label] = np_truth_box[np_truth_cls == label]

        preds_per_idx.append(preds_per_label)
        truth_per_idx.append(truth_per_label)

    ap = {}
    for label in range(min_label, n_classes):
        true_positives = []
        scores = []
        for b in range(batch_size):
            preds = preds_per_idx[b][label]
            truth = truth_per_idx[b][label]
            detected_annotations = []
            for pred_idx in range(preds.shape[0]):
                if truth.shape[0] == 0:
                    true_positives.append(0)
                    continue
                conf = preds[pred_idx][4]
                scores.append(conf)
                overlaps = bbox_iou_numpy(np.expand_dims(preds[pred_idx][:4], axis=0), truth)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_thres and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sort by score
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        metrics['AP[{}]'.format(label)] = average_precision
    metrics['mAP'] = np.mean([metrics['AP[{}]'.format(l)] for l in range(min_label, n_classes)])
    return metrics


def test(args, model, loader, prefix='', verbose=True):
    
    print("train: Beginning test")

    loss_fn = loss_fns.DetectorLossFn().to(args.device)
    metrics = defaultdict(list)
    with torch.no_grad():
        for (img, det) in tqdm(loader):
            batch_size = img.shape[0]
            img = img.to(args.device).expand(batch_size, 3, *(img.shape[2:]))
            det = det.to(args.device)
            det_hat = model(img)
            loss, loss_dict = loss_fn(det_hat[0], det_hat[1], det)
            metrics['loss'].append(loss.item())
            for k, v in loss_dict.items():
                metrics[k].append(np.mean([v]))
            for k, v in get_metrics(det, det_hat).items():
                metrics[k].append(v)

        for k in replace_metric_by_mean:
            metrics[k] = np.mean(metrics[k])

        # Print!
        if verbose:
            start_string = '#### {} evaluation ####'.format(prefix)
            print(start_string)
            for k, v in metrics.items():
                print('#### {} = {}'.format(k, v[-1]))
            print(''.join(['#' for _ in range(len(start_string))]))
    return metrics


def train_detector_v0(args, dataset, train_loader, test_loader):
    output_dir = args.base_output
    model = load_model(args)
    if args.cuda:
        model = model.cuda()
    if args.dataparallel:
        model = nn.DataParallel(model)

    loss_fn = loss_fns.DetectorLossFn().to(args.device)

    params = list(model.parameters())
    optimizer = optim.Adam(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    metrics = defaultdict(list)
    
    print("train: Beginning train")

    for epoch_idx in range(args.epochs):
        print('Starting epoch {}'.format(epoch_idx))
        epoch_metrics = defaultdict(list)
        tic = time.time()
        for (img, det) in tqdm(train_loader):
            batch_size = img.shape[0]
            img = img.to(args.device).expand(batch_size, 3, *(img.shape[2:]))
            det = det.to(args.device)
            det_hat = model(img)
            loss, loss_dict = loss_fn(det_hat[0], det_hat[1], det)

            epoch_metrics['loss'].append(loss.item())
            for k, v in loss_dict.items():
                epoch_metrics[k].append(np.mean([v]))
            for k, v in get_metrics(det, det_hat).items():
                epoch_metrics[k].append(v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for k, v in epoch_metrics.items():
            metrics[k].append(np.mean(v))

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
    return model, metrics


def run_v0(args):

    print("main: Creating dataset")
        
    dataset = custom_dataset.VesicleDetectionDataset(args.data_path, load_from_pickle=(not args.load_data), cpu_count=args.num_workers)
    if args.load_data:    
        dataset.dump_to_pickle('data/vesicle_dataset.pkl')
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
    model, metrics = train_detector_v0(args, dataset, train_loader, test_loader)
    return model, metrics


if __name__ == '__main__':
    args = get_args()
    general_utils.dump_everything(args)
    model, metrics = run_v0(args)
    with open(os.path.join(args.base_output, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
