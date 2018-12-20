import os, sys
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import numpy as np
import torch

class dataset_multi(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot

    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    # B
    images_C = os.listdir(os.path.join(self.dataroot, opts.phase + 'C'))
    self.C = [os.path.join(self.dataroot, opts.phase + 'C', x) for x in images_C]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.C_size = len(self.C)
    self.dataset_size = max(self.A_size, self.B_size, self.C_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b
    self.input_dim_C = opts.input_dim_c

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d, C: %d images'%(self.A_size, self.B_size, self.C_size))
    return

  def __getitem__(self, index):
    cls = random.randint(0,2)
    c_org = np.zeros((3,))
    c_trg = np.zeros((3,))
    if cls == 0:
      data = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      c_org[0] = 1
      c_trg[random.choice([1,2])] = 1
    elif cls == 1:
      data = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
      c_org[1] = 1
      c_trg[random.choice([0,2])] = 1
    else:
      data = self.load_img(self.C[random.randint(0, self.C_size - 1)], self.input_dim_C)
      c_org[2] = 1
      c_trg[random.choice([0,1])] = 1
    return data, torch.FloatTensor(c_org), torch.FloatTensor(c_trg)

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size


class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.size = len(self.img)
    self.input_dim = input_dim

    self.c_org = np.zeros((3,))
    if setname == 'A':
      self.c_org[0] = 1
    elif setname == 'B':
      self.c_org[1] = 1
    elif setname == 'C':
      self.c_org[2] = 1
    else:
      print('UNKNOWN SETNAME')
      sys.exit()

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img, self.c_org

  def __len__(self):
    return self.size
