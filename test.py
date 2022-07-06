import os
import sys
import time
import torch
import torchvision
import numpy as np
from PIL import Image
sys.path.append(os.path.abspath("."))
from dataloader import ImgDataset
from model import Ranking, ToPatches
from utils import ConvertYcbcr2Bgr, ConvertBgr2Ycbcr, Resize

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Ranking(device=device).to(device)
    model.load_state_dict(torch.load("weights/best_ranking.pth", map_location=device))
    model.eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop([720,1280]),
        Resize(1./4, 1./4),
        ConvertBgr2Ycbcr(),
        torchvision.transforms.ToTensor()
    ])
    img = Image.open("DIV2K_valid_HR/0890.png")
    input_tensor = transform(img)
    input_tensor = input_tensor.to(torch.float32).to(device)
    input_tensor = ToPatches((20, 20))(input_tensor[0,:,:].unsqueeze(0).unsqueeze(0))

    input_tensor = torch.nn.Upsample(size=(20,20))(input_tensor)
    rank = model(input_tensor)
    indices = torch.argsort(rank, dim=0).squeeze(1)

    test_img = torchvision.transforms.CenterCrop([720,1280])(img)
    np_img = np.array(test_img)
    out_img = np.zeros((720, 1280, 3))
    for i, idx in enumerate(indices):
        idx = int(idx)
        x = idx // 16
        y = idx % 16
        out_img[x*80:x*80+80,y*80:y*80+80,:] = np_img[x*80:x*80+80,y*80:y*80+80,:]/(i/40+1)
    image = Image.fromarray(np.uint8(out_img))
    image.save('test.png')