'''
Implement datasets
'''

import os
import time
import multiprocessing
import pickle
import numpy as np
from skimage import draw
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from joblib import Parallel, delayed


vesicle_codes = {'docked_DCV': 0, 'LV': 1, 'tethered_SV': 2, 'CCV': 3, 'docked_SV': 4, 'DCV': 5, 'SV': 6}  # noqa


def load_image(path):
    return np.array(Image.open(path))


def load_segmentation(path, img=None):
    ''' If img is not None, fills up pixels with appropriate value
    '''
    df = pd.read_csv(path, sep='\t', header=None)
    if img is not None:
        for i in range(df.shape[0]):
            if df.iloc[i][1] in vesicle_codes:
                val = vesicle_codes[df.iloc[i][1]]
                # x, y, r are not the same as r, c, radius...
                img[draw.circle(df.iloc[3], df.iloc[2], df.iloc[4])] = val
            else:
                pass  # other methods have not been implemented
    return df, img


class PixelSegmentationDataset(TorchDataset):
    def __init__(self, image_list, window_size=128, preload=True):
        '''
        image_list: Either a list of tuples of segmentation files, image files
            or a path to a pickle containing said list.
        '''
        if isinstance(image_list, str):
            with open(image_list, 'rb') as f:
                image_list = pickle.load(f)
        self.seg_paths, self.img_paths = zip(*image_list)

        self.images = [load_image(path) for path in self.img_paths]
        segs = [np.zeros(img.shape) for img in self.images]
        self.segs = [load_segmentation(self.seg_paths[i], img=segs[i])[1] for i in range(len(segs))]

    def __len__(self):
        return self.n
