import os
import cv2
from scipy.io import loadmat
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np 
import random
from PIL import Image 
import pandas as pd 




batch_w = 600
batch_h = 400

class NTIRELoader(torch.utils.data.Dataset):
    def __init__(self,  img_dir, opt):
        self.img_dir = img_dir
        # self.gt_img_dir = gt_img_dir
        # self.task = task
        self.opt = opt 
        self.scenes_list = os.listdir(self.img_dir)
        print(self.scenes_list)
        # self.annotations = pd.read_csv(csv_file)
        # self.train_low_data_names = []
        # self.train_gt_data_names = []

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        data_list = []

        self.input_list = []
        self.gt_list = []

        # for scene in os.listdir(img_dir):
        #     for image in os.listdir(os.path.join(self.img_dir + scene + '/inp/')):
        #         self.input_list += [os.path.join(self.img_dir + scene + '/inp/' + image )]

        for image in os.listdir(os.path.join(self.img_dir + '/train_inp/')):
            self.input_list += [os.path.join(self.img_dir + '/train_inp/' + image)]

        for image in os.listdir(os.path.join(self.img_dir + '/train_gt/')):
            self.gt_list += [os.path.join(self.img_dir + '/train_gt/' + image)]
        

        # for scene in os.listdir(img_dir):
        #     for image in os.listdir(os.path.join(img_dir + scene + '/gt/')):
        #         self.gt_list += [os.path.join(img_dir + scene + '/gt/' + image )]
                
        # self.gt_list = [gt_img.replace("/train_inp/", "/train_gt/") for gt_img in self.input_list]

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        im = im.resize((2200, 2000))
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):
        inp = self.load_images_transform(self.input_list[index])
        gt = self.load_images_transform(self.gt_list[index])


        h = inp.shape[0]
        w = inp.shape[1]

        if self.opt.crop:
            if self.opt.crop_size > 0:
                h, w = gt.shape[:2]
                rand_h = random.randint(0, h - self.opt.crop_size)
                rand_w = random.randint(0, w - self.opt.crop_size)
                gt = gt[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]    # (256, 256, 31), in range [0, 1], float64
                inp = inp[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]    # (256, 256, 3), in range [0, 1], float64'


        gt = torch.from_numpy(gt.astype(np.float32).transpose(2, 0, 1)).contiguous()
        inp = torch.from_numpy(inp.astype(np.float32).transpose(2, 0, 1)).contiguous()

        img_name = self.input_list[index].split('\\')[-1]

        return inp, gt, img_name


    def __len__(self):
        return len(self.input_list)




class NTIRELoaderCV2(torch.utils.data.Dataset):
    def __init__(self,  img_dir, opt):
        self.img_dir = img_dir
        # self.gt_img_dir = gt_img_dir
        # self.task = task
        self.opt = opt 
        self.scenes_list = os.listdir(self.img_dir)
        print(self.scenes_list)
        # self.annotations = pd.read_csv(csv_file)
        # self.train_low_data_names = []
        # self.train_gt_data_names = []

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        data_list = []

        self.input_list = []
        self.gt_list = []

        # for scene in os.listdir(img_dir):
        #     for image in os.listdir(os.path.join(self.img_dir + scene + '/inp/')):
        #         self.input_list += [os.path.join(self.img_dir + scene + '/inp/' + image )]

        for image in os.listdir(os.path.join(self.img_dir + '/train_inp/')):
            self.input_list += [os.path.join(self.img_dir + '/train_inp/' + image)]

        for image in os.listdir(os.path.join(self.img_dir + '/train_gt/')):
            self.gt_list += [os.path.join(self.img_dir + '/train_gt/' + image)]
        
        self.input_list.sort()
        self.gt_list.sort()
        

        # for scene in os.listdir(img_dir):
        #     for image in os.listdir(os.path.join(img_dir + scene + '/gt/')):
        #         self.gt_list += [os.path.join(img_dir + scene + '/gt/' + image )]
                
        # self.gt_list = [gt_img.replace("/train_inp/", "/train_gt/") for gt_img in self.input_list]

    def load_images_transform(self, file):
        # im = Image.open(file).convert('RGB')
        im = cv2.imread(file)
        # im = im.resize((batch_w, batch_h))
        im = cv2.resize(im, (batch_w, batch_h))
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):
        print(self.input_list[index], self.gt_list[index])
        inp = self.load_images_transform(self.input_list[index])
        gt = self.load_images_transform(self.gt_list[index])


        h = inp.shape[0]
        w = inp.shape[1]

        if self.opt.crop:
            if self.opt.crop_size > 0:
                h, w = gt.shape[:2]
                rand_h = random.randint(0, h - self.opt.crop_size)
                rand_w = random.randint(0, w - self.opt.crop_size)
                gt = gt[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]    # (256, 256, 31), in range [0, 1], float64
                inp = inp[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]    # (256, 256, 3), in range [0, 1], float64'


        gt = torch.from_numpy(gt.astype(np.float32).transpose(2, 0, 1)).contiguous()
        inp = torch.from_numpy(inp.astype(np.float32).transpose(2, 0, 1)).contiguous()

        img_name = self.input_list[index].split('\\')[-1]

        return inp, gt, img_name


    def __len__(self):
        return len(self.input_list)

batch_w = 600
batch_h = 400
# data loader
class DataloaderSupervised(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_img_dir, csv_file, task, opt):
        self.low_img_dir = img_dir
        self.gt_img_dir = gt_img_dir
        self.task = task
        self.opt = opt
        self.annotations = pd.read_csv(csv_file)
        self.train_low_data_names = []
        self.train_gt_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))


        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        # im = Image.open(file).convert('RGB')
        im = cv2.imread(file)
        # im = im.resize((batch_w, batch_h))
        im = cv2.resize(im, (batch_w, batch_h))
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):

        inp = self.load_images_transform(os.path.join(self.low_img_dir, self.annotations.iloc[index, 0]))
        gt = self.load_images_transform(os.path.join(self.gt_img_dir, self.annotations.iloc[index, 1]))


        h = inp.shape[0]
        w = inp.shape[1]

        if self.opt.crop:
            if self.opt.crop_size > 0:
                h, w = gt.shape[:2]
                rand_h = random.randint(0, h - self.opt.crop_size)
                rand_w = random.randint(0, w - self.opt.crop_size)
                gt = gt[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]    # (256, 256, 31), in range [0, 1], float64
                inp = inp[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]    # (256, 256, 3), in range [0, 1], float64'


        gt = torch.from_numpy(gt.astype(np.float32).transpose(2, 0, 1)).contiguous()
        inp = torch.from_numpy(inp.astype(np.float32).transpose(2, 0, 1)).contiguous()

        # img_name = self.input_list[index].split('\\')[-1]
        img_name = os.path.join(self.low_img_dir, self.annotations.iloc[index, 1]).split('\\')[-1]


        return inp, gt, img_name

    def __len__(self):
        return len(self.annotations)