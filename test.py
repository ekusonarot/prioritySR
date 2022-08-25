import os
import sys
import time
import cv2
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
    device = "cpu"
    model = Ranking(device=device).to(device)
    model.load_state_dict(torch.load("weights/ranking_epoch99_loss0.32774006202816963.pth", map_location=device))
    model.eval()
    for i in model.parameters():
      i.requires_grad=False

    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop([720,1280]),
        Resize(1./4, 1./4),
        torchvision.transforms.ToTensor()
    ])
    img = Image.open("DIV2K_valid_HR/0890.png")
    input_tensor = transform(img)
    input_tensor = input_tensor.to(torch.float32).to(device)
    input_tensor = ToPatches((20, 20),padding=(5,5,5,5))(input_tensor[:,:,:].unsqueeze(0))

    start = time.perf_counter()
    rank = model(input_tensor)
    print("{}\n".format(time.perf_counter()-start))
    indices = torch.argsort(rank, dim=0, descending=False).squeeze(1)

    test_img = torchvision.transforms.CenterCrop([720,1280])(img)
    np_img = np.array(test_img)
    out_img = np.zeros((720, 1280, 3))
    for i, idx in enumerate(indices):
        idx = int(idx)
        x = idx // 16
        y = idx % 16
        #out_img[x*80:x*80+80,y*80:y*80+80,:] = np_img[x*80:x*80+80,y*80:y*80+80,:]/(i/30+1)
        #'''
        if i < 70:
            out_img[x*80:x*80+80,y*80:y*80+80,:] = np_img[x*80:x*80+80,y*80:y*80+80,:]
        else:
            out_img[x*80:x*80+80,y*80:y*80+80,:] = np_img[x*80:x*80+80,y*80:y*80+80,:]/2
        #'''
    image = Image.fromarray(np.uint8(out_img))
    image.save('cnn.png')

    finder = cv2.FastFeatureDetector_create()
    w_index = 16
    h_index = 9
    filenum = w_index*h_index
    block_size = 80
    start = time.perf_counter()
    keylist = [len(finder.detect(np_img[index%(h_index*w_index)//w_index*block_size:index%(h_index*w_index)//w_index*block_size+block_size,index%(h_index*w_index)%w_index*block_size:index%(h_index*w_index)%w_index*block_size+block_size,:]))
        for index in range(filenum)
    ]
    print("{}\n".format(time.perf_counter()-start))
    np_keylist = np.array(keylist)
    ranklist = np.argsort(-np_keylist)
    for i, idx in enumerate(ranklist):
        idx = int(idx)
        x = idx // 16
        y = idx % 16
        #out_img[x*80:x*80+80,y*80:y*80+80,:] = np_img[x*80:x*80+80,y*80:y*80+80,:]/(i/30+1)
        #'''
        if i < 70:
            out_img[x*80:x*80+80,y*80:y*80+80,:] = np_img[x*80:x*80+80,y*80:y*80+80,:]
        else:
            out_img[x*80:x*80+80,y*80:y*80+80,:] = np_img[x*80:x*80+80,y*80:y*80+80,:]/2
        #'''
    image = Image.fromarray(np.uint8(out_img))
    image.save('corner.png')