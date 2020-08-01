# Automated Detection/Segmentation of Ultrastructure Features in Electron Microscope Images of Neuronal Synapses

## Requirements:
- Anaconda
- Python 3.5+
- PyTorch (1.0)
- PIL
- skimage

## Geting the Data
1. Download original dataset automated_vesicle_detection.zip from Box link (contact for details).
2. Download the trained models file models.zip from https://drive.google.com/file/d/1RQrN0kBqgm5Ig01iWplbPOMPGmBSWfyX/view?usp=sharing

## Creating the Datasets
1. From base directory, run `python -m data.make_image_list -b [folder containing automated_vesicle_detection/] -o [output file]`. This creates a pickled list of image paths.
2. Run `python -m utils.dataset -o [file to dump processed dataset]` to speed up data loading. It creates a npz file that has two arrays - 'images', 'segs' which are windows

## TODO

1. Properly compare 2-class models for each class vs. multi-class model. This involves creating aggregated images from output patches i.e. for each pixel of each original image, aggregate predictions from every patch of that image by averaging or majority vote.
2. 
