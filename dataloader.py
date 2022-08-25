import sys
import os
import glob
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
sys.path.append(os.path.abspath("."))
from utils import ConvertBgr2Y, Resize

class ImgDataset(Dataset):
  def __init__(self, input_img_dir1, input_img_dir2=None, target_img_dir=None, transform1=None, transform2=None, target_transform=None):
    self.input_img_dir1 = input_img_dir1
    self.input_img_dir2 = input_img_dir2
    self.target_img_dir = target_img_dir
    if self.input_img_dir1:
      self.input_img_paths1 = glob.glob(self.input_img_dir1+"/*")
    if self.input_img_dir2:
      self.input_img_paths2 = glob.glob(self.input_img_dir2+"/*")
    if self.target_img_dir:
      self.target_img_paths = glob.glob(self.target_img_dir+"/*")
    self.transform1 = transform1
    self.transform2 = transform2
    self.target_transform = target_transform
  
  def __len__(self):
    return len(self.input_img_paths1)

  def __getitem__(self, idx):
    input_image1 = Image.open(self.input_img_paths1[idx])
    input_image2 = Image.open(self.input_img_paths2[idx])
    target_image = Image.open(self.target_img_paths[idx])
    if self.transform1:
      input_image1 = self.transform1(input_image1)
    if self.transform2:
      input_image2 = self.transform2(input_image2)
    if self.target_transform:
      target_image = self.target_transform(target_image)
    return input_image1, input_image2, target_image
    

if __name__ == "__main__":
  transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop([1000,2000]),
    Resize(0.5, 0.5),
    ConvertBgr2Y(),
    torchvision.transforms.ToTensor()
  ])
  target_transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop([1000,2000]),
    ConvertBgr2Y(),
    torchvision.transforms.ToTensor()
  ])
  dataset = ImgDataset("DIV2K_train_HR", "DIV2K_train_HR", transform, transform)
  dataloader = DataLoader(dataset, batch_size = 30, shuffle = True)
  for input_image, target_image in dataloader:
    print(input_image.shape)