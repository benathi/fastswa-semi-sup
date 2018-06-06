import re
import os
import pickle
import sys

from tqdm import tqdm
from torchvision.datasets import CIFAR100
import matplotlib.image
import numpy as np

work_dir_path = sys.argv[1] # workdir
images_dir_path = sys.argv[2] # images/cifar/cifar100/by-image/

work_dir = os.path.abspath(work_dir_path)
test_dir = os.path.abspath(os.path.join(images_dir_path, 'test'))
train_dir = os.path.abspath(os.path.join(images_dir_path, 'train+val'))

cifar100 = CIFAR100(work_dir, download=True)


def load_file(file_name):
    with open(os.path.join(work_dir, cifar100.base_folder, file_name), 'rb') as meta_f:
        return pickle.load(meta_f, encoding="latin1")


def unpack_data_file(source_file_name, target_dir, start_idx):
    print("Unpacking {} to {}".format(source_file_name, target_dir))
    data = load_file(source_file_name)
    #print("data keys", data.keys()) #dict_keys(['filenames', 'batch_label', 'fine_labels', 'coarse_labels', 'data'])
    for idx, (image_data, label_idx) in tqdm(enumerate(zip(data['data'], data['fine_labels'])), total=len(data['data'])):
        subdir = os.path.join(target_dir, label_names[label_idx])
        name = "{}_{}.png".format(start_idx + idx, label_names[label_idx])
        os.makedirs(subdir, exist_ok=True)
        image = np.moveaxis(image_data.reshape(3, 32, 32), 0, 2)
        matplotlib.image.imsave(os.path.join(subdir, name), image)
    return len(data['data'])


#label_names = load_file('meta')['label_names']
#print(load_file('meta').keys()) # dict_keys(['fine_label_names', 'coarse_label_names'])
label_names = load_file('meta')['fine_label_names'] 
print("Found {} label names: {}".format(len(label_names), ", ".join(label_names)))

start_idx = 0
for source_file_path, _ in cifar100.test_list:
    start_idx += unpack_data_file(source_file_path, test_dir, start_idx)

start_idx = 0
for source_file_path, _ in cifar100.train_list:
    start_idx += unpack_data_file(source_file_path, train_dir, start_idx)
