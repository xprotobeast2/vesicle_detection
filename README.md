Automating data analysis of slice of neuron-tissue-things?

# Requirements:
- Anaconda
- Python 3.5+
- PyTorch (1.0)
- PIL
- skimage


# Data
1. Download from Box (contact for details)
2. From base directory, run `python -m data.make_image_list -b [folder containing automated_vesicle_detection/] -o [output file]`
3. Run `python -m utils.dataset -o [file to dump processed dataset]` to speed up data loading. It creates a npz file that has two arrays - 'images', 'segs' which are windows
