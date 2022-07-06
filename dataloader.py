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
  def __init__(self, input_img_dir, target_img_dir, transform=None, target_transform=None):
    self.input_img_dir = input_img_dir
    self.target_img_dir = target_img_dir
    self.input_img_paths = glob.glob(self.input_img_dir+"/*")
    self.target_img_paths = glob.glob(self.target_img_dir+"/*")
    self.transform = transform
    self.target_transform = target_transform
  
  def __len__(self):
    return len(self.input_img_paths)

  def __getitem__(self, idx):
    input_image = Image.open(self.input_img_paths[idx])
    target_image = Image.open(self.target_img_paths[idx])
    if self.transform:
      input_image = self.transform(input_image)
    if self.target_transform:
      target_image = self.target_transform(target_image)
    return input_image, target_image
    

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