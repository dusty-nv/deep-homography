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


def pred_homography(batch, model_file, img_width, img_height, scale=32):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print('running on device ' + str(device))

  # normalize the input
  batch = normalize_batch(batch).to(device)

  print('\nnormalized_batch:')
  print(batch)
  print(batch.size())

  net = HomographyNet(img_width, img_height)
  net.load_state_dict(torch.load(model_file))  
  net.eval().to(device)

  with torch.no_grad():
    raw_output = net(batch)
    print('\nraw_output:')
    print(raw_output)
    output = raw_output.detach().cpu().numpy()*scale
    print('\noutput:')
    print(output)
    mean_shift = np.mean(output, axis=0)
    print('\nmean_shift:')
    print(output)

    pts1 = np.float32([0, 0, patch_size, 0, patch_size, patch_size, 0, patch_size]).reshape(-1,1,2)
    pts2 = mean_shift.reshape(-1,1,2) + pts1
   
    print('\npts1:')
    print(pts1)
    print('\npts2:')
    print(pts2)

    cv_h = cv2.findHomography(pts1, pts2)[0]
    print('\nocv homography:')
    print(cv_h)

    h = np.linalg.inv(cv_h)
    
    print('\ninv homography:')
    print(h)

    return h


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='deep_homography.pytorch')
  parser.add_argument('--patch_size', default=128, type=int, help='image patch size to resize to')
  parser.add_argument('--img_width', default=128, type=int, help='image patch size to resize to (network input)')
  parser.add_argument('--img_height', default=128, type=int, help='image patch size to resize to (network input)')
  parser.add_argument('--i', default=-1, type=int, help='number of image pair in training set')
  parser.add_argument('--file1', default='', type=str, help='file of image 1')  
  parser.add_argument('--file2', default='', type=str, help='file of image 2')  
  args = parser.parse_args(sys.argv[1:])

  patch_size = args.patch_size

  if args.i > -1: 
    i = args.i
    f1, f2 = 'data/synth_data/{:09d}_orig.jpg'.format(i), 'data/synth_data/{:09d}_warp.jpg'.format(i)
  elif args.file1 != '' and args.file1 != '':
    f1, f2 = args.file1, args.file2
  else:
    raise ValueError('specify file name pair or integer for image pair in training data')

  # load the input data
  img1, img2, batch = prep_batch(f1, f2, args.img_width, args.img_height)

  # run the inference model
  h = pred_homography(batch, args.model, args.img_width, args.img_height)

  # warp the input by the homography
  new_img = cv2.warpPerspective(img1, h, (args.img_width, args.img_height))

  # visualize the results
  plt.figure(1)
  plt.subplot(131)
  plt.imshow(img1, cmap='Greys_r')
  plt.title('original img')
  plt.subplot(133)
  plt.imshow(new_img, cmap='Greys_r')
  plt.title('orig img warped \n by pred homography')
  plt.subplot(132)
  plt.imshow(img2, cmap='Greys_r')
  plt.title('"warped" img')
  plt.show()

