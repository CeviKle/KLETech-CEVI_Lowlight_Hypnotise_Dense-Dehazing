import argparse
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import os
import GPUtil
import network
from PIL import Image

from resblock_256 import resblock,conv_relu_res_relu_block


# csv_file="/home/cvg-ws05/msi_up/LLIE/codebase/data_final.csv"
root_directory_input="/home/pegasus/DATA/Nikhil/LLIE/denoising_data/SIDD/test/INPUT/"
# root_directory_ground_truth="/home/cvg-ws05/msi_up/LLIE/data/"
modelPath = "/home/pegasus/Experiments/Nikhil/denoising/ABL/SIDD_bs4/Train-20221214-203426/model_epochs/weights_100.pt"
inputlist = os.listdir(root_directory_input)
print(inputlist)
destinationPath = "/home/pegasus/Experiments/Nikhil/denoising/ABL/SIDD_bs4/results/"
if(os.path.exists(destinationPath) == False):
    os.mkdir(destinationPath)


def normalise(img, target_type_min, target_type_max, target_type ):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

model = resblock(conv_relu_res_relu_block, 16, 3,3) 
model.eval()
model.load_state_dict(torch.load(modelPath,map_location=torch.device('cuda')))
print("Model is initialised")

for i in tqdm(range(len(inputlist))):
    image =cv2.resize( cv2.imread(root_directory_input + inputlist[i]),(1024,1024))
#    image = Image.open(root_directory_input + inputlist[i]).resize((1024,1024))
    
    #size = image.shape

    image = torch.from_numpy(np.float32(image/255.0)).unsqueeze(0)

    # image = torch.from_numpy(image)

    #print(image.shape)
    

    image = image.permute(0, 3, 1, 2)

    print("sending images to model.........")
    outputImage = model(image)

    outputImage = outputImage.permute(0,2,3,1)

    outputImage = outputImage[0].data.cpu().float().numpy()
    outputImage = cv2.resize(outputImage, (600, 400))
    print("saving image.....")
    cv2.imwrite(os.path.join(destinationPath + inputlist[i]),outputImage*255)

