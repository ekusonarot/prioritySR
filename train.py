import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
sys.path.append(os.path.abspath("."))
from fsrcnn import FSRCNN
from utils import ConvertBgr2Y, Resize
from dataloader import ImgDataset
from model import RankSR, CustomLoss, ToPatches

epoch = 100
lr = 1e-2
scale_factor = 4
batch_size = 20

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = RankSR(scale_factor=scale_factor, device=device).to(device)
  criterion = CustomLoss(device=device)
  optimizer = optim.Adam([
    {"params" : model.ranking.conv2d_1.parameters()},
    {"params" : model.ranking.conv2d_2.parameters(), "lr" : lr*0.1},
    {"params" : model.ranking.dense.parameters(), "lr" : lr*0.01}
    ], lr=lr)
  
  transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop([720,1280]),
    Resize(1./scale_factor, 1./scale_factor),
    ConvertBgr2Y(),
    torchvision.transforms.ToTensor()
  ])
  target_transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop([720,1280]),
    ConvertBgr2Y(),
    torchvision.transforms.ToTensor()
  ])
  train_set = ImgDataset("DIV2K_train_HR", "DIV2K_train_HR", transform, target_transform)
  train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
  valid_set = ImgDataset("DIV2K_valid_HR", "DIV2K_valid_HR", transform, target_transform)
  valid_loader = DataLoader(valid_set, batch_size = batch_size, shuffle = False)
  
  print('Training set has {} instances'.format(len(train_set)))
  print('Validation set has {} instances'.format(len(valid_set)))
  toPatchesX = ToPatches((720//scale_factor//9, 1280//scale_factor//16))
  toPatchesY = ToPatches((720//9, 1280//16))
  for e in range(epoch):
    total_loss = 0.
    for s, data in enumerate(train_loader):
      x_image, y_image = data
      x_image = x_image.to(device, dtype=torch.float)
      y_image = y_image.to(device, dtype=torch.float)
      x_image = toPatchesX(x_image)
      y_image = toPatchesY(y_image)
      optimizer.zero_grad()

      rank, fsrcnn, bicubic = model(x_image, is_train=True)
      loss = criterion(rank, fsrcnn, bicubic, y_image)
      loss.backward()

      optimizer.step()
      running_loss = loss.item()
      bar = s*batch_size*20//len(train_set)
      total_loss += running_loss
      print("\r[train] epoch{}[{}] loss:{}".format(e, '='*bar+'-'*(20-bar), running_loss, rank[:5,0]), end="")
      del loss
    print("\n[train] total loss {}\n".format(total_loss))
    total_loss = 0.
    for s, data in enumerate(valid_loader):
      x_image, y_image = data
      x_image = x_image.to(device, dtype=torch.float)
      y_image = y_image.to(device, dtype=torch.float)
      x_image = toPatchesX(x_image)
      y_image = toPatchesY(y_image)
      rank, fsrcnn, bicubic = model(x_image, is_train=True)

      loss = criterion(rank, fsrcnn, bicubic, y_image)

      running_loss = loss.item()
      bar = s*batch_size*20//len(valid_set)
      total_loss += running_loss
      print("\r[valid] epoch{}[{}] loss:{}".format(e, '='*bar+'-'*(20-bar), running_loss), end="")
      del loss
    print("\n[valid] total loss {}\n".format(total_loss))
    torch.save(model.ranking.state_dict(), "weights/ranking_epoch{}_loss{}.pth".format(e, total_loss))
    

