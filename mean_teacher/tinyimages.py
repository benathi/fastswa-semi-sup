import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
from PIL import Image

class TinyImages(Dataset):
  """ Tiny Images Dataset """
  def __init__(self, which, transform=None,
    pkl_path='data-local/tiny_index.pkl',
    meta_path='data-local/cifar100_meta.meta',
    NO_LABEL = -1):

    assert which in ['237k', '500k', 'tiny_all'], 'Invalid options'
    with open(pkl_path, 'rb') as f:
      tinyimg_index = pickle.load(f)
    if which == '237k':
        print("Using all classes common with CIFAR-100.")
        with open(meta_path, 'rb') as f:
            cifar_labels = pickle.load(f)['fine_label_names']
        cifar_to_tinyimg = { 'maple_tree': 'maple', 'aquarium_fish' : 'fish' }
        cifar_labels = [l if l not in cifar_to_tinyimg else cifar_to_tinyimg[l] for l in cifar_labels]
        load_indices = sum([list(range(*tinyimg_index[label])) for label in cifar_labels], [])
    elif which == '500k':
        print("Using {} random images.".format(which))
        idxs_filename = os.path.join("/".join(meta_path.split("/")[:-1]), 'tiny_idxs500k.pkl')
        if os.path.isfile(idxs_filename):
          print("Loading indices from file", idxs_filename)
          load_indices = pickle.load(open(idxs_filename, 'rb'))
        else:
          print("Saving indices to file", idxs_filename)
          num_all_images = max(e for s, e in tinyimg_index.values())
          load_indices = np.arange(num_all_images)
          np.random.shuffle(load_indices)
          load_indices = load_indices[:500000] # note: we need to fix this 
          load_indices.sort()
          pickle.dump(load_indices, open(idxs_filename, 'wb'))
          print("Saved")
    elif which == 'tiny_all':
      print("Using All Tiny Images")
      num_all_images = max(e for s, e in tinyimg_index.values())
      load_indices = np.arange(num_all_images)
      np.random.shuffle(load_indices)
    # now we have load_indices
    self.indices = load_indices
    self.len = len(self.indices)
    self.transform = transform
    self.no_label = NO_LABEL
    print("Length of the dataset = {}".format(self.len))

  def __len__(self):
    return self.len

  def __getitem__(self, idx, verbose=False):
    if verbose: print("tiny to idx = {} actual idx = {}".format(idx, self.indices[idx]))
    img = self.load_tiny_image(self.indices[idx])
    if self.transform:
      img = self.transform(img)
    return img, self.no_label

  def load_tiny_image(self, idx, data_path='data-local/images/cifar/cifar_tiny_images/tiny_images.bin'):
    img = None
    with open(data_path, 'rb') as f:
      f.seek(3072 * idx)
      img = np.fromfile(f, dtype='uint8', count=3072).reshape(3, 32, 32).transpose((0, 2, 1))
      img = Image.fromarray(np.rollaxis(img, 0, 3))
    return img


if __name__ == "__main__":
  c = TinyImages()
  