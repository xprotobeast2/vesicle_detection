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

From base directory, run the following command:

`python -m data.make_image_list -b [folder containing automated_vesicle_detection/] -o [output file]` 

This creates a pickled list of image paths. Feed this into utils/dataset.py with:

`python -m utils.dataset [OPTIONS] -i [input image list] -o [dataset name]`

You may use `python -m utils.dataset -h` to print the list of options and their descriptions. The above command will create a dataset.
For example, 

## Training a model

You can run `main_v0.py` with something like the following command:

`python main_v0.py [OPTIONS] --data_path=[dataset name]`

For example,

`python main_v0.py --cuda -dp --loss_func=dice -p 256 -s 128 -b 16 -e 50 -r eg_model_name -ud 4 -nc 3 --data_path=eg_dataset`

The above will train a 3-class segmentation model on a dataset called `eg_dataset` with patch size 256 and overlap 128 with a UNet type model with depth 4. The training is done for 50 epochs with batch size 16 with soft dice as the loss function. If a gpu is available it will be used, and the model data loading will use DataParallel. The model checkpoints will be saved to `./output/seg/eg_model_name`.

## Models

In `modules/unet.py`, there are descriptions of the two types of Pytorch models used so far for segmentation. As the code currently stands,  `unet.FRUNet` is the class that is being used, and is hardcoded in the call to `load_model` in `main_v0.py`. This may be parametrized in the future. 

## TODO
- Regenerate datasets, some segmentation ground truths don't seem to match up with the images sometimes. Verify this.
- Properly compare 2-class models for each class vs. multi-class model. This involves creating aggregated images from output patches i.e. for each pixel of each original image, aggregate predictions from every patch of that image by averaging or majority vote.
- Try to improve segmentation test dice scores. Particularly the upstream dice-score defined in the .ipynb file.
- The output must be a file just like the input .txt files located in `analysis_files` directory of `automatic_vesicle_detection.zip`. 
- Turn this into a usable, deployable tool that can take input images and output detections as .txt files of the defined format (check the `load_segmentation` function in `utils/dataset.py`).
