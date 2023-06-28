import torch
import numpy as np
import pandas as pd
import json
import cv2
import pickle
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import torchvision
from CTImageQuality.models.edcnn import EDCNN


# load images

lables_path = 'data/LDCTIQAG2023_train/train.json'
images_dir = 'data/LDCTIQAG2023_train/image'

def load_images(data):

    image_np = list()
    for name,val in data.items():
        img = cv2.imread(f'{images_dir}/{name}', cv2.IMREAD_UNCHANGED)
        image_np.append(img)
    return np.stack(image_np)

with open(lables_path) as file:
    all_lables = pd.Series(json.load(file))

all_images = load_images(all_lables)
train_sz = 800
train_lables, test_lables = all_lables[:train_sz], all_lables[train_sz:]
train_images, test_images = all_images[:train_sz], all_images[train_sz:]

class PairDS(torch.utils.data.IterableDataset):
    def __init__(self, images, lables):
        self.images = images
        self.lables = lables

    def __iter__(self):
        self.unused = np.arange(0, self.images.shape[0])
        np.random.shuffle(self.unused)
        self.unused = iter(np.reshape(self.unused,(-1,2)))
        return self.generator()
    
    def generator(self):
        for ind1, ind2 in self.unused:
            diff = self.lables.iloc[ind1] - self.lables.iloc[ind2]

            yield self.images[ind1][np.newaxis], self.images[ind2][np.newaxis], diff

class HydraModel(nn.Module):
    def __init__(self, backbone, vector_size=1000):
        super().__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(vector_size*2, 200) 
        self.fc2 = nn.Linear(200, 1) 
    
    def forward(self, x):
        im1, im2 = x
        vec1 = self.backbone(im1)
        vec2 = self.backbone(im2)
        
        combined = torch.cat((vec1, vec2),dim=1)
        x = self.fc1(combined)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        
        return x


back = torchvision.models.resnet18()
back.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
edcnn = EDCNN()
weigths = torch.load('weights/edcnn.pkl', map_location=torch.device("cuda"))
edcnn.load_state_dict(weigths)
nnet = HydraModel(back)#cnn)
nnet.train()
nnet.cuda()
pds = PairDS(all_images, all_lables)
lld = DataLoader(pds,batch_size=10)
optimizer = torch.optim.AdamW(nnet.parameters(), lr=0.001)



for epoch in range(10):
    log = tqdm(enumerate(lld))
    torch.save(nnet.state_dict(), 'ghidra_point.pth')
    for i, data in log:
        model_tensor = next(nnet.parameters())
        im1, im2, target = (x.to(model_tensor) for x in data)
        pred = nnet((im1,im2))
        loss = F.mse_loss(pred.squeeze(), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        itm = loss.item()
        if i==0:
            losser = itm
        losser = losser*0.9+itm*0.1
        log.set_postfix({"loss": losser})



















