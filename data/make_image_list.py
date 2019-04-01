import os
import sys
import glob
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--base', default='/data/cellseg/data/',)
parser.add_argument('-o', '--output_filename', default='image_list.pkl', help='')
args = parser.parse_args()


def cr_code_converter(code):
    return '_'.join(code.split('_')[::-1])


seg2img = {}

base_data_dir = os.path.join(args.base, 'automated_vesicle_detection')
analysis_dir = 'analysis_files'
for folder in glob.glob('{}/{}/*'.format(base_data_dir, analysis_dir)):
    id2seg = {}
    id2img = {}
    seg_cr_code = os.path.split(folder)[1]
    img_cr_code = cr_code_converter(seg_cr_code)
    for seg_file in glob.glob(folder + '/**', recursive=True):
        if seg_file.endswith('.txt'):
            id2seg[os.path.split(seg_file)[1][:-4]] = seg_file
    for img_file in (glob.glob(os.path.join(base_data_dir, img_cr_code) + "/**", recursive=True)):
        if img_file.lower().endswith('.tif'):
            id2img[os.path.split(img_file)[1][:-4]] = img_file
    for i, s in id2seg.items():
        if i in id2img:
            seg2img[s] = id2img[i]
    print('{}: {} seg, {} img'.format(folder, len(id2seg), len(id2img)))
print('Total {} files'.format(len(seg2img)))

dumpable = []
for segf, imgf in seg2img.items():
    dumpable.append((segf, imgf))
print('Dumping to {}'.format(os.path.join(args.base, args.output_filename)))

with open(os.path.join(args.base, args.output_filename), 'wb') as f:
    pickle.dump(dumpable, f)
