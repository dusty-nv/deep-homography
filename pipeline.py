import struct
import os
import urllib
import zipfile
import pickle
import PIL
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def jpg_reader(filename):
  img = PIL.Image.open(filename)
  return np.asarray(img)


class Normalize:
  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    image = image.transpose((2,0,1))
    image = 2*(image/255) - 1
    label = label/32

    return {'image': torch.from_numpy(image),
            'label': torch.from_numpy(label)}


class HomographyDataset(Dataset):
  def __init__(self, image_path, label_path, val_frac=0.05, mode='train'): 
    self.image_path = image_path
    self.label_path = label_path

    print('image mode = ' + mode)
    print('image path = ' + image_path)
    print('label path = ' + label_path)

    assert os.path.isdir(image_path), 'Dataset is missing'
    assert os.path.isfile(label_path), 'Label file is missing'

    self.transforms = transforms.Compose([Normalize()])
  
    with open(label_path, 'r') as f:
      num_and_label = [line.rstrip().rstrip(',').split(';') for line in f]

    #print(num_and_label)

    L = len(num_and_label)
    idx = int(val_frac*L)
    
    if mode == 'train':
      self.num_and_label = num_and_label[idx:]
    elif mode == 'eval':
      self.num_and_label = num_and_label[:idx]
    else:
      raise ValueError('no such mode')
      
  def __len__(self):
    return len(self.num_and_label)

  def __getitem__(self, idx):
    input_file_orig = '{:s}/{:s}'.format(self.image_path, self.num_and_label[idx][0])
    input_file_warp = '{:s}/{:s}'.format(self.image_path, self.num_and_label[idx][1])

    img_orig = jpg_reader(input_file_orig)
    img_warp = jpg_reader(input_file_warp)

    img = np.concatenate([img_orig[:,:,None], img_warp[:,:,None]], axis=2).astype(np.float32)
   
    label_str = self.num_and_label[idx][2] 
    label = np.array([float(el) for el in label_str.split(',')]).astype(np.float32)

    sample = {
      'image': img,
      'label': label}
    sample = self.transforms(sample)
    return sample


if __name__ == '__main__':
  dataset = HomographyDataset()
  dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4)

  for i_batch, sample_batch in enumerate(dataloader):
    print(sample_batch['image'].size())
    print(sample_batch['label'].size())
