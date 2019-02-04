import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import HomographyNet


def normalize_batch(image_batch):
  image_batch = image_batch.transpose((0,3,1,2))
  image_batch = 2*(image_batch/255) - 1
  return torch.from_numpy(image_batch)


def read_and_convert(f):
  img = cv2.imread(f)
  assert img is not None, 'no such file'
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

  
def prep_batch(f1, f2, img_width, img_height):

  img1 = read_and_convert(f1)
  img2 = read_and_convert(f2)

  patch1 = cv2.resize(img1, (img_width, img_height))
  patch2 = cv2.resize(img2, (img_width, img_height))

  batch = np.concatenate([patch1[:,:,None], patch2[:,:,None]], axis=2).astype(np.float32)[None, :, :, :]

  return img1, img2, batch


def export_onnx(batch, model_in, model_out, img_width, img_height, scale=32):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print('running on device ' + str(device))

  batch = normalize_batch(batch).to(device)

  net = HomographyNet(img_width, img_height)
  net.load_state_dict(torch.load(model_in))  
  net.eval().to(device)
  
  input_names = [ "input_0" ]
  output_names = [ "output_0" ]

  torch.onnx.export(net, batch, model_out, verbose=True, input_names=input_names, output_names=output_names)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_in', type=str, default='deep_homography.pytorch')
  parser.add_argument('--model_out', type=str, default='deep_homography.onnx')
  parser.add_argument('--img_width', default=128, type=int, help='image patch size to resize to (network input)')
  parser.add_argument('--img_height', default=128, type=int, help='image patch size to resize to (network input)')
  parser.add_argument('--i', default=-1, type=int, help='number of image pair in training set')
  parser.add_argument('--file1', default='', type=str, help='file of image 1')  
  parser.add_argument('--file2', default='', type=str, help='file of image 2')  
  args = parser.parse_args(sys.argv[1:])

  if args.i > -1: 
    i = args.i
    f1, f2 = 'data/synth_data/{:09d}_orig.jpg'.format(i), 'data/synth_data/{:09d}_warp.jpg'.format(i)

  elif args.file1 != '' and args.file1 != '':
    f1, f2 = args.file1, args.file2

  else:
    raise ValueError('specify file name pair or integer for image pair in training data')

  img1, img2, batch = prep_batch(f1, f2, args.img_width, args.img_height)
  export_onnx(batch, args.model_in, args.model_out, args.img_width, args.img_height)
  
