'''
Implement datasets
'''

import os
import sys
import time
import multiprocessing
import pickle
import h5py
import numpy as np
import pandas as pd
import skimage
import cv2
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from joblib import Parallel, delayed
from tqdm import tqdm

vesicle_codes = {'docked_DCV': 1, 'LV': 2, 'tethered_SV': 3, 'CCV': 4, 'docked_SV': 5, 'DCV': 6, 'SV': 7, 'Plasma_membrane': 8}  # noqa


DTYPE = np.uint8
IM_SHAPE = (1024,1024)

def load_image(path):
    """
    Loads a single image from disk using PIL. 

    params: path is the location of the image on disk

    return: the image as a numpy array of type DTYPE. 
    """
    return np.array(Image.open(path)).astype(DTYPE)

    

def load_segmentation(path, shape=None):
    """
    A very important function that constructs the segmentation map for 
    and image in the dataset. In the dataset, look at the .txt files within
    the analysis_files directory to understand how this function works. The 
    final output of the entire system should be a .txt file similar to those. 

    The text file is formatted as : 1st column (tab) 2nd column (tab) 3rd column (tab)
    4th column (tab) 5th column (tab) 6th column. The description of each column is as follows:

        1st: the tool_type used in imageJ / Fiji. 1 is straight line. 
             3 is freehand selection.  7 is freehand line.

        2nd: name of object such as plasma membrane, active zone, SV (synaptic vesicles), 
             endosomes, LV (large vesicles), and DCV (dense core vesicles). 
             For SV close to active sone, there are more categories: docked or tethered SV.

        3rd: If the object is either plasma membrane or active zone, 
             it reports perimeter or length. If the object is endosomes, 
             it reports area. Otherwise, this is skipped.

        4th : x-coordinates. For tool-type 3 and 7, there will be multiple x-coordinates, 
                each will be separated by “,”. For tool-type 1, it will give a single point
                from the center of the object (i.e. the center of a synaptic vesicle).

        5th: y- coordinates.

        6th: Only if tool type is 1. this is diameter.

    params: path is the location of the image on disk, shape is the dimensions of
            the segmentation map. 

    return: the image as a numpy array of type DTYPE. 
    """

    df = pd.read_csv(path, sep='\t', header=None)
    if shape is not None:
        img = np.zeros(shape, dtype=np.float32)
        for i in range(df.shape[0]):
            if df.iloc[i][1] in vesicle_codes:  
                val = vesicle_codes[df.iloc[i][1]]
                if int(df.iloc[i][0]) == 1:
                    if args.vesicle:
                        # x, y, r are not the same as r, c, radius...
                        x, y, r = (float(df.iloc[i][2]), float(df.iloc[i][3]), float(df.iloc[i][4])/2)
                        img[skimage.draw.circle(y, x, r, shape=img.shape)] = val
                else:
                    if args.membrane:
                        
                        # 
                        x = np.array(list(map(int, df.iloc[i][3].split(',')))).astype(int)
                        y = np.array(list(map(int, df.iloc[i][4].split(',')))).astype(int)
                        
                        y[y>=img.shape[0]] = img.shape[0] - 1
                        x[x>=img.shape[1]] = img.shape[1] - 1

                        pts = np.vstack([x,y]).T.astype('int32')
                        cv2.polylines(img, [pts], False, val, thickness=10)
            else:
                pass  # other methods have not been implemented

    else:
        img = None
    return df.loc[df[1].isin(vesicle_codes) & df[0].isin(['1'])], img


def get_non_zero_indices(seg, min_frac):
    """
    A helper function that allows filtering of patches that do not contain
    important objects.

    params: seg is a single segmentation map viewed as windows. min_frac allows
            removal of all windows that aren't at least min_frac foreground class.

    return: the indices of the patches that have at least min_frac foreground cells. 
    """
    return np.where(np.reshape(seg > 0, (seg.shape[0], -1)).sum(1) > (min_frac * np.prod(seg.shape[1:])))


class PixelSegmentationDataset(TorchDataset):
    def __init__(self, image_list, window_size=128, step_size=64, 
        pad_mode='symmetric', cpu_count=10, minimum_patch_density=0.001, n_classes=2, 
        max_size=-1, normalize=False, transform=None):
        """
        Create a pytorch dataset from the dataset of EM images of neuronal synapses.
        This class can be invoked in 3 ways based on the image_list parameter.

        params: 

            image_list: Can be a path to 1) a pickle file 2) a compressed numpy zip 
                        (.npz) file containing images and segmentations 3) a directory 
                        containing the images and segmentations as separate .npy files
            window_size: The dimension of the square image patch that will be used to 
                         train the model.
            step_size: The overlap between adjacent patches or the step taken when viewing 
                       an image as windows.
            pad_mode: The padding mode used with skimage.view_as_windows
            cpu_count: The number of cpus used during window construction
            minimum_patch_density: the minimum amount of pixels in the patched segmentation
                                   that need to be a foreground class in order to keep the 
                                   patch for training/testing.
            n_classes: the number of types of objects in the image to be identified.
            max_size: A limit on the dataset size
            normalize: Translate the dataset to 0 mean and unit standard deviation
            transform: the pytorch transforms that need to be applied to the dataset. 
        """    
        super().__init__()
        tic = time.time()
        self.window_size = window_size
        self.step_size = step_size
        self.pad_mode = pad_mode
        self.n_classes = n_classes
        self.normalize = normalize
        self.transform = transform

        # Load dataset from disk
        if isinstance(image_list, str):
            if image_list.endswith('.pkl'):
                self.load_from_image_list(image_list, window_size=window_size, step_size=step_size, pad_mode=pad_mode, cpu_count=cpu_count, minimum_patch_density=minimum_patch_density, max_size=max_size)
            elif image_list.endswith('.npz'):
                self.load_from_npz(image_list)
            else:
                # Pass in a directory to store npys
                self.load_from_npy(image_list)
        
        # Show class weights
        print('Class weights: %s'%str(self.class_weights))
        print('Loaded data in {:.1f}s'.format(time.time() - tic))

    def load_from_npz(self, image_list):
        """
        Loads the dataset from disk when it is a numpy zipped file.
        This function uses read-only memory mapping.

        params: image_list is the path to the .npz file
        """
        npz = np.load(image_list, mmap_mode='r')
        self.images = npz['images']
        self.segs = npz['segs']
        self.class_weights = npz['weights']
        self.n = self.images.shape[0]

    def dump_to_npy(self, npy_dir):
        """
        Saves the dataset to disk to a directory with 
        the images and segs as .npy files.
        This is the best method when the dataset is large.

        params: npy_dir is the path to the dataset directory
        """
        os.makedirs(npy_dir, exist_ok=True)
        np.save(npy_dir+'/images.npy', self.images)
        np.save(npy_dir+'/segs.npy', self.segs)
        np.save(npy_dir+'/weights.npy', self.class_weights)

    def load_from_npy(self, npy_dir):
        """
        Loads the dataset from disk from .npy files.
        Only method that works on a low RAM machine 
        for the whole datasets. This function uses read-only 
        memory mapping.

        params: npy_dir is the path to the dataset directory containing
                images.npy,segs.npy and weights.npy
        """
        self.images = np.load(npy_dir+'/images.npy', mmap_mode='r')
        self.segs = np.load(npy_dir+'/segs.npy', mmap_mode='r')
        self.class_weights = np.load(npy_dir+'/weights.npy')
        self.n = self.images.shape[0]

    def dump_to_npz(self, file):
        """
        Save the dataset as a compressed numpy zip file. This works
        well when the dataset size is smaller.

        params: npy_dir is the path to the dataset directory containing
                images.npy,segs.npy and weights.npy
        """
        np.savez_compressed(file, images=self.images, segs=self.segs, weights=self.class_weights)

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
        
        tic = time.time()
        
        if args.load_images:
            print("========== Loading data... ==========")


            # TODO Parallelize loading
            imgs_h5 = h5py.File('data/images.hdf5', mode='w')
            segs_h5 = h5py.File('data/segmentations.hdf5', mode='w')
            for i in tqdm(range(len(self.img_paths))):
                img = load_image(self.img_paths[i])
                seg = load_segmentation(self.seg_paths[i], img.shape)[1]
                #img = Image.fromarray(curr_seg).resize(IM_SHAPE)
                imgs_h5.create_dataset('%d'%i, data=img)
                segs_h5.create_dataset('%d'%i, data=seg)   
            print("Total Images: %d"%len(imgs_h5.keys()))
            print("Total Segs: %d"%len(segs_h5.keys()))
            
            imgs_h5.close()
            segs_h5.close()

            print('[{:.2f}s] Loaded images'.format(time.time() - tic))

        # # create hdf5 file in data folder
        if args.filter_patches:
            print("========== Windowing data... ==========")

            # Process into images
            imgs_h5 = h5py.File('data/images.hdf5', mode='r')
            segs_h5 = h5py.File('data/segmentations.hdf5', mode='r')
            windowed_imgs_h5 = h5py.File('data/windowed_images.hdf5', mode='w')
            windowed_segs_h5 = h5py.File('data/windowed_segmentations.hdf5', mode='w')
            for i in tqdm(range(len(self.img_paths))):
                windowed_img = np.reshape(window_fn(pad_fn(imgs_h5.get('%d'%i))), (-1, window_size, window_size))            
                windowed_seg = np.reshape(window_fn(pad_fn(segs_h5.get('%d'%i))), (-1, window_size, window_size))
                idxs = get_non_zero_indices(windowed_seg, minimum_patch_density)
                windowed_imgs_h5.create_dataset('%d'%i, data=windowed_img[idxs])   
                windowed_segs_h5.create_dataset('%d'%i, data=windowed_seg[idxs])   
            print("Total Processed Images: %d"%len(windowed_imgs_h5.keys()))
            windowed_imgs_h5.close()
            windowed_segs_h5.close()
            imgs_h5.close()
            segs_h5.close()

        print("========== Concatenating data... ==========")

        windowed_imgs_h5 = h5py.File('data/windowed_images.hdf5', mode='r')
        windowed_segs_h5 = h5py.File('data/windowed_segmentations.hdf5', mode='r')
        self.images = []
        self.segs = []
        for i in tqdm(range(len(self.img_paths))):
            img = np.array(windowed_imgs_h5.get('%d'%i))
            seg = np.array(windowed_segs_h5.get('%d'%i))
            #print(i, img.shape, seg.shape)
            self.images.append(img)
            self.segs.append(seg)
        windowed_imgs_h5.close()
        windowed_segs_h5.close()
        self.images = np.concatenate(self.images)
        self.images = self.images.astype(np.float32)/255
        if self.normalize:
            self.images = (self.images - self.images.mean(0)) / (self.images.std(0) + 1e-8)
        print('[{:.2f}s] Concatenated images'.format(time.time() - tic))
        self.segs = np.concatenate(self.segs)
        uniques, counts = np.unique(self.segs, return_counts=True)
        self.class_weights = (counts*1.0)/counts.sum()
        print('[{:.2f}s] Concatenated segs'.format(time.time() - tic))
        self.n = self.images.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # TODO Data augmentation
        img = np.array(self.images[idx])
        seg = np.array(self.segs[idx])
        if self.transform:
            img = self.transform(Image.fromarray(np.reshape(img, (self.window_size, self.window_size))))
            seg = self.transform(Image.fromarray(seg))
        else:
            img = np.reshape(img, (1, self.window_size, self.window_size))
        return (img, seg)

class VesicleDetectionDataset(TorchDataset):
    def __init__(self, image_list, load_from_pickle=None, window_size=256, cpu_count=10, n_classes=1, max_objects=80, classify_vesicle_type=False):
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
            }, f, protocol=4)

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
                                x - 2*r, y - 2*r, x + 2*r, y + 2*r
                            ]))
                    return window.astype(np.float32), np.stack(det_windows).astype(np.float32)
                else:
                    return None
            return _fn

        images = []
        dets = []

        for i in tqdm(range(len(image_list))):

            seg_path, img_path = image_list[i]
            
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
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default='data/image_list.pkl', help='Location of image list pickle')
    parser.add_argument('-o', '--output_file', default='/data/cellseg/data/all_data.npz', help='Location to dump dataset')
    parser.add_argument('-p', '--window_size', type=int, default=128, help='Window size for dataset')
    parser.add_argument('-s', '--step_size', type=int, default=64, help='Step size for windows')
    parser.add_argument('-d', '--patch_density', type=float, default=0.0, help='Ignores patches that have less than this percentage of foreground class')
    parser.add_argument('-l', '--load_images', action='store_true', help='Do we need to reload images and segmentations')
    parser.add_argument('-f', '--filter_patches', action='store_true', help='Do we need to refilter and window images')
    parser.add_argument('-z', '--save_compressed', action='store_true', help='Dump dataset to npz')
    parser.add_argument('-y', '--save_array', action='store_true', help='Dump dataset to npy directory')
    parser.add_argument('--vesicle', action='store_true', help='Include vesicle class')
    parser.add_argument('--membrane', action='store_true', help='Include membrane class')
    parser.add_argument('--normalize', action='store_true', help='Center dataset to 0 and scale dataset to 1')


    
    args = parser.parse_args()

    import time; tic = time.time()
    dataset = PixelSegmentationDataset(args.input_file, cpu_count=multiprocessing.cpu_count() // 3, window_size=args.window_size, step_size=args.step_size, minimum_patch_density=args.patch_density, normalize=args.normalize)
    print('Loading took {}s'.format(time.time() - tic))

    tic = time.time()
    if args.save_compressed:    
        dataset.dump_to_npz(args.output_file)
    elif args.save_array:
        dataset.dump_to_npy(args.output_file)
    print('Dumping images took {}s'.format(time.time() - tic))
    tic = time.time()
    dataset = PixelSegmentationDataset(args.output_file)
    print('Loading from npz took {}s'.format(time.time() - tic))