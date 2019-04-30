'''
Implement datasets
'''

import os
import time
import multiprocessing
import pickle
import numpy as np
import pandas as pd
import skimage
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from joblib import Parallel, delayed


vesicle_codes = {'docked_DCV': 1, 'LV': 2, 'tethered_SV': 3, 'CCV': 4, 'docked_SV': 5, 'DCV': 6, 'SV': 7}  # noqa


DTYPE = np.float32


def load_image(path):
    return np.array(Image.open(path)).astype(DTYPE)


def load_segmentation(path, shape=None):
    ''' If img is not None, fills up pixels with appropriate value
    '''
    df = pd.read_csv(path, sep='\t', header=None)
    if shape is not None:
        img = np.zeros(shape, dtype=np.float32)
        for i in range(df.shape[0]):
            if df.iloc[i][1] in vesicle_codes:
                if int(df.iloc[i][0]) == 1:
                    val = vesicle_codes[df.iloc[i][1]]
                    # x, y, r are not the same as r, c, radius...
                    x, y, r = (float(df.iloc[i][2]), float(df.iloc[i][3]), float(df.iloc[i][4]) / 2)
                    img[skimage.draw.circle(y, x, r, shape=img.shape)] = val
                else:
                    # TODO handle straight line/free hand tool type things
                    pass
            else:
                pass  # other methods have not been implemented

    else:
        img = None
    return df.loc[df[1].isin(vesicle_codes) & df[0].isin(['1'])], img


def get_non_zero_indices(segs, min_frac):
    idxs = []
    for i in range(len(segs)):
        inner_idx = np.where(np.reshape(segs[i] > 0, (segs[i].shape[0], -1)).sum(1) > (min_frac * np.prod(segs[i].shape)))
        idxs.append((i, inner_idx))
    return idxs


class PixelSegmentationDataset(TorchDataset):
    def __init__(self, image_list, load_from_npz=None, window_size=128, step_size=64, pad_mode='symmetric', cpu_count=10, minimum_patch_density=0.0001, n_classes=2, max_size=-1):
        '''
        image_list: Either a list of tuples of segmentation files, image files
            or a path to a pickle containing said list.

        load_from_npz: None, False or True.
            If None then set to True if image_list is string ending in .npz
        image_list:
            One of:
            1. list of (seg_file, image_file) tuples or path to pickled file containing list
            2. npz file that can be loaded using np.load

        Other args: Useful only when loading from actual list of images or file (not NPZ)
        '''
        super().__init__()
        tic = time.time()
        self.window_size = window_size
        self.step_size = step_size
        self.pad_mode = pad_mode
        self.n_classes = n_classes
        if load_from_npz is None:
            load_from_npz = isinstance(image_list, str) and image_list.endswith('.npz')
        if not load_from_npz:
            self.load_from_image_list(image_list, window_size=window_size, step_size=step_size, pad_mode=pad_mode, cpu_count=cpu_count, minimum_patch_density=minimum_patch_density, max_size=max_size)
        else:
            self.load_from_npz(image_list)
        self.images /= 255
        self.images = (self.images - self.images.mean(0)) / (self.images.std(0) + 1e-8)
        print('Loaded data in {:.1f}s'.format(time.time() - tic))

    def load_from_npz(self, image_list):
        npz = np.load(image_list)
        self.images = npz['images']
        self.segs = npz['segs']
        self.n = self.images.shape[0]

    def dump_to_npz(self, file):
        np.savez_compressed(file, images=self.images, segs=self.segs)

    def load_from_image_list(self, image_list, window_size=128, step_size=64, pad_mode='symmetric', cpu_count=10, minimum_patch_density=0.0001, max_size=-1):
        if isinstance(image_list, str):
            with open(image_list, 'rb') as f:
                image_list = pickle.load(f)

        def pad_fn(img):
            return skimage.util.pad(
                img,
                (
                    (0, (window_size - (img.shape[0] % window_size))),
                    (0, (window_size - (img.shape[1] % window_size)))
                ),
                self.pad_mode
            )

        def window_fn(img):
            return skimage.util.view_as_windows(img, window_size, step_size)
        # Load images

        self.seg_paths, self.img_paths = zip(*image_list)
        if max_size > 0:
            self.img_paths = self.img_paths[:max_size]
            self.seg_paths = self.seg_paths[:max_size]
        # TODO Parallelize loading
        tic = time.time()
        self.images = [load_image(path) for path in self.img_paths]
        print('[{:.2f}s] Loaded images'.format(time.time() - tic))

        def _load(a, b):
            return load_segmentation(a, shape=b)[1]

        self.segs = Parallel(n_jobs=cpu_count)(delayed(_load)(self.seg_paths[i], self.images[i].shape) for i in range(len(self.images)))
        print('[{:.2f}s] Loaded segmentations'.format(time.time() - tic))

        # Process into images
        self.images = [np.reshape(window_fn(pad_fn(img)), (-1, window_size, window_size)) for img in self.images]
        print('[{:.2f}s] Windowed images'.format(time.time() - tic))
        self.segs = [np.reshape(window_fn(pad_fn(seg)), (-1, window_size, window_size)) for seg in self.segs]
        print('[{:.2f}s] Windowed segmentations'.format(time.time() - tic))

        acceptable_indices = get_non_zero_indices(self.segs, minimum_patch_density)
        print('[{:.2f}] Found {} patches to use'.format(time.time() - tic, sum(len(idxs[0]) for _, idxs in acceptable_indices)))
        for i, idxs in acceptable_indices:
            self.images[i] = self.images[i][idxs]
            self.segs[i] = self.segs[i][idxs]
        print('[{:.2f}s] Filtered out unnecessary images'.format(time.time() - tic))
        self.images = np.concatenate(self.images)
        print('[{:.2f}s] Concatenated images'.format(time.time() - tic))
        self.segs = np.concatenate(self.segs)
        print('[{:.2f}s] Concatenated segs'.format(time.time() - tic))
        self.n = self.images.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # TODO Data augmentation
        return (
            np.reshape(self.images[idx], (1, self.window_size, self.window_size)),
            self.segs[idx].astype(np.int64)
        )


class VesicleDetectionDataset(TorchDataset):
    def __init__(self, image_list, load_from_pickle=None, window_size=416, cpu_count=10, n_classes=1, max_objects=80, classify_vesicle_type=False):
        ''' Each datapoint is an (img, det) tuple where
            det is an array whose rows represent
            [nclass, top-left-x, top-left-y, bottom-right-x, bottom-right-y]
        '''
        super().__init__()
        tic = time.time()
        self.classify_vesicle_type = classify_vesicle_type
        self.max_objects = max_objects
        self.window_size = window_size
        self.n_classes = n_classes
        if load_from_pickle is None:
            load_from_pickle = False  # isinstance(image_list, str) and image_list.endswith('.pkl')
        if load_from_pickle:
            self.load_from_pickle(image_list)
        else:
            self.load_from_image_list(image_list, window_size=window_size, cpu_count=cpu_count)
        self.images /= 255
        self.images = (self.images - self.images.mean(0)) / (self.images.std(0) + 1e-8)
        print('Loaded data in {:.1f}s'.format(time.time() - tic))

    def load_from_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            objs = pickle.load(f)
        self.images = objs['images']
        self.dets = objs['dets']
        self.n = self.images.shape[0]

    def dump_to_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'images': self.images,
                'dets': self.dets,
            })

    def load_from_image_list(self, image_list, window_size=416, cpu_count=10):
        if isinstance(image_list, str):
            with open(image_list, 'rb') as f:
                image_list = pickle.load(f)

        self.seg_paths, self.img_paths = zip(*image_list)

        def window_function_constructor(img, df):
            def _fn(xi, yi):
                det_windows = []
                min_x = xi * window_size
                max_x = min_x + window_size
                min_y = yi * window_size
                max_y = min_y + window_size
                detections = df.loc[
                    ((df[2].astype(np.float32) >= min_x) & (df[2].astype(np.float32) < max_x) & (df[3].astype(np.float32) >= min_y) & (df[3].astype(np.float32) < max_y))
                ]
                if detections.shape[0] > 0:
                    window = img[min_x: max_x, min_y: max_y]
                    for i in range(detections.shape[0]):
                        x, y, r = (float(detections.iloc[i][2]), float(detections.iloc[i][3]), float(detections.iloc[i][4]) / 2)
                        x = x - min_x
                        y = y - min_y
                        det_windows.append(
                            np.array([
                                vesicle_codes[detections.iloc[i][1]] if self.classify_vesicle_type else 0,
                                x - r, y - r, x + r, y + r
                            ]))
                    return window.astype(np.float32), np.stack(det_windows).astype(np.float32)
                else:
                    return None
            return _fn

        images = []
        dets = []

        for seg_path, img_path in image_list:
            # Load image and segmentation
            full_img = load_image(img_path)
            df, _ = load_segmentation(seg_path, shape=None)  # Only need dataframe

            nx = full_img.shape[0] // window_size
            ny = full_img.shape[1] // window_size

            window_fn = window_function_constructor(full_img, df)
            # Get windows, and for each window, figure out objects
            zipped_windows = Parallel(n_jobs=cpu_count)(
                delayed(window_fn)(xi, yi)
                for xi in range(nx)
                for yi in range(ny)
            )
            zipped_windows = [k for k in zipped_windows if k is not None]
            if len(zipped_windows) > 0:
                windows, targets = zip(*(zipped_windows))
                images.append(windows)
                dets.extend(targets)

        self.images = np.concatenate(images, axis=0)
        self.dets = dets  # A list.
        self.n = self.images.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.dets[idx].shape[0] < self.max_objects:
            dets = np.pad(self.dets[idx], ((0, self.max_objects - self.dets[idx].shape[0]), (0, 0)), 'constant')
        else:
            dets = self.dets[idx][:self.max_objects, :]
        return np.reshape(self.images[idx], (1, self.window_size, self.window_size)), dets


if __name__ == '__main__':
    with open('/data/cellseg/data/image_list.pkl', 'rb') as f:
        imagelist = pickle.load(f)
    import time; tic = time.time()
    dataset = PixelSegmentationDataset(imagelist, cpu_count=multiprocessing.cpu_count() // 3)
    print('Loading took {}s'.format(time.time() - tic))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_file', default='/data/cellseg/data/all_data.npz', help='Location to dump dataset')
    args = parser.parse_args()
    tic = time.time()
    dataset.dump_to_npz(args.output_file)
    print('Dumping images took {}s'.format(time.time() - tic))
    tic = time.time()
    dataset = PixelSegmentationDataset(args.output_file, load_from_npz=True)
    print('Loading from npz took {}s'.format(time.time() - tic))
