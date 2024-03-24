from vainF_ssim import MS_SSIM
from utils import Vgg16, run_test
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
import cv2
import warnings
import shutil
import os
import json
import argparse
import logging
import time
import utils
from PIL import Image
import glob 
import sys
import random
import torchvision.transforms as transforms
import torch.nn as nn



CUDA_LAUNCH_BLOCKING = 1

from resblock_256 import resblock,conv_relu_res_relu_block


warnings.filterwarnings("ignore")

print("[INFO] Libraries loaded...")




with open('configs.json', 'r') as f:
    cfg = json.load(f)
print("[INFO] configs loaded...")

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--version", help='experiments version')
parser.add_argument("-ct", "--continue_train", help="resume training if abruptly stopped", default=0)
parser.add_argument("-d", "--dataset",help="Dataset to train on", default='lol')
parser.add_argument('--n_resblocks', type=int, default=16,help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,help='residual scaling')
parser.add_argument('--scale', type=str, default=2,help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=256,help='output patch size')
parser.add_argument('--n_colors', type=int, default=3,help='number of input color channels to use')
parser.add_argument('--o_colors', type=int, default=3,help='number of output color channels to use')
parser.add_argument("-bs", "--batch_size", type=int, default=8, help='defines batch size for training')
parser.add_argument("-e", "--epochs", type=int, default=130, help='defines training epochs')
parser.add_argument('-lr',"--learning_rate", type=float, default=0.0003, help='learning rate')
parser.add_argument('-a',"--alpha", type=float, default=0.5, help='alpha value for comb. loss')
parser.add_argument('-b',"--beta", type=float, default=0.5, help='beta value for comb. loss')
parser.add_argument('-g',"--gamma", type=float, default=0.5, help='gamma value for comb. loss')
parser.add_argument("-o", "--save", type=str, default='ABL', help='outputs path')


args = parser.parse_args()


args.save = args.save + '/' + args.version + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
image_path_full_video = args.save + '/image_epochs_full_video/'

os.makedirs(image_path, exist_ok=True)
os.makedirs(image_path_full_video, exist_ok=True)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

start = time.time()
logging.info("start time = %s", start)
device = torch.device("cuda")

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

if not torch.cuda.is_available():
    logging.info('No GPU device available')
    sys.exit(1)

logging.info("args = %s", args)

batch_w = 600
batch_h = 400
# data loader
class DataloaderSupervised(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_img_dir, csv_file, task):
        self.low_img_dir = img_dir
        self.gt_img_dir = gt_img_dir
        self.task = task
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
        im = Image.open(file).convert('RGB')
        im = im.resize((600, 400))
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):

        low = self.load_images_transform(os.path.join(self.low_img_dir, self.annotations.iloc[index, 1]))
        gt = self.load_images_transform(os.path.join(self.gt_img_dir, self.annotations.iloc[index, 0]))


        h = low.shape[0]
        w = low.shape[1]
        #
        h_offset = random.randint(0, max(0, h - batch_h - 1))
        w_offset = random.randint(0, max(0, w - batch_w - 1))
        #
        # if self.task != 'test':
        #     low = low[h_offset:h_offset + batch_h, w_offset:w_offset + batch_w]

        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))

        gt = np.asarray(gt, dtype=np.float32)
        gt = np.transpose(gt[:, :, :], (2, 0, 1))

        # img_name = self.train_low_data_names[index].split('\\')[-1]
        img_name = os.path.join(self.low_img_dir, self.annotations.iloc[index, 1]).split('\\')[-1]

        # if self.task == 'test':
        #     # img_name = self.train_low_data_names[index].split('\\')[-1]
        #     return torch.from_numpy(low), img_name

        return torch.from_numpy(low), torch.from_numpy(gt), img_name

    def __len__(self):
        return len(self.annotations)


root = os.getcwd()
os.chdir(root)

# load data


if args.dataset == 'ntire':
    # NTIRE 2022 Dataset
    dataset = DataloaderSupervised(csv_file="/home/pegasus/DATA/Nikhil/LLIE/data_final.csv",
                      img_dir="/home/pegasus/DATA/Nikhil/LLIE/ntire",
                      gt_img_dir="/home/pegasus/DATA/Nikhil/LLIE/ntire",
                      task='train')
    train_data, test_data = torch.utils.data.random_split(dataset, [520, 50])
elif args.dataset == 'lol':
    # LOLDataset
    train_data = DataloaderSupervised(csv_file="/home/pegasus/DATA/Nikhil/LLIE/denoising_data/SIDD/SIDD.csv",
                         img_dir="/home/pegasus/DATA/Nikhil/LLIE/denoising_data/SIDD/INPUT/",
                         gt_img_dir="/home/pegasus/DATA/Nikhil/LLIE/denoising_data/SIDD/GT/",
                      task='train')
    test_data = DataloaderSupervised(csv_file="/home/pegasus/DATA/Nikhil/LLIE/denoising_data/SIDD/SIDD_test.csv",
                        img_dir="/home/pegasus/DATA/Nikhil/LLIE/denoising_data/SIDD/INPUT/",
                        gt_img_dir="/home/pegasus/DATA/Nikhil/LLIE/denoising_data/SIDD/GT/",
                      task='test')
    test_data_full = DataloaderSupervised(csv_file="/home/pegasus/DATA/Nikhil/LLIE/denoising_data/SIDD/SIDD_test.csv",
                        img_dir="/home/pegasus/DATA/Nikhil/LLIE/denoising_data/SIDD/INPUT/",
                        gt_img_dir="/home/pegasus/DATA/Nikhil/LLIE/denoising_data/SIDD/GT/",
                      task='test')

else:
    print("[INFO] Dataset not specified....")
print("[INFO] Data loaded...")
# print(len(dataset))

# spilt train and test data


train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=4, shuffle=True, drop_last=True, num_workers=4)
test_loader_full = torch.utils.data.DataLoader(
    test_data_full, batch_size=4, shuffle=True, drop_last=True, num_workers=4)

device = torch.device("cuda")
model = resblock(conv_relu_res_relu_block, args.n_resblocks, 3,3) 

if torch.cuda.device_count()>1:
   print("No. of gpu's used = ",torch.cuda.device_count())
   model = nn.DataParallel(model)
model = model.to(device)


print(model)
print('Trainable parameters : {}\n'.format(sum(p.numel()
      for p in model.parameters() if p.requires_grad)))



print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))

iter_num = 0

# losses and optimizers
l1_loss = torch.nn.L1Loss()
feature_loss = Vgg16().to(device)

dssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


optimizer.zero_grad()
loss_iter_list = ['Iteration']
iter_LR = ['Iter_LR']

genLoss = 0.0
discLoss = 0.0
genTrainLoss = 0.0
discTrainLoss = 0.0
print("[INFO] Model configs loaded...")
total_step = 0

idx = 0
for epoch in range(args.epochs):
        model.train()
        losses = []

        loop = tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size), desc="Epoch: {}".format(epoch+1))
        for idx, (inp, gt, _) in loop:
            # idx+=1
            total_step += 1
            inp = inp.type(torch.FloatTensor)
            gt = gt.type(torch.FloatTensor)
            inp = Variable(inp, requires_grad=False).to(device)
            gt = Variable(gt, requires_grad=False).to(device)

            optimizer.zero_grad()
            # loss = model._loss(inp)
            pred = model(inp)
            
            
            # i_list, en_list, in_list, _ = pred
            loss_l1 = l1_loss(pred, gt)
            loss_fl = feature_loss(pred, gt, which='relu2')
            loss_ssim = 1- dssim(pred, gt)
            loss = args.alpha*loss_l1 + args.beta * loss_fl + args.gamma * loss_ssim 
            loss.backward()
            optimizer.step()


            losses.append(loss.item())
            # logging.info('train-epoch %03d %03d %f', epoch, idx, loss)
            utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))

            if epoch % 10 == 0 and epoch <75 and total_step != 0:
                # logging.info('train %03d %f', epoch, loss)
                model.eval()
                with torch.no_grad():
                    for _, (input, gt, image_name) in enumerate(test_loader):
                        input = Variable(input, volatile=True).cuda()
                        # gt = Variable(gt, volatile=True).cuda()

                        image_name = image_name[0].split('/')[-1].split('.')[0]

                        pred = model(input)
                

                        u_name = '%s.png' % (image_name + '_restored_' + str(epoch))
                        
                        u_path = image_path + '/' + u_name
                        # saving intermediate outputs
                        o1_name = '%s.png' % (image_name + '_GT_' + str(epoch))
                        o1_path = image_path + '/' + o1_name

                        o3_name = '%s.png' % (image_name + '_gt_' + str(epoch))
                        o3_path = image_path + '/' + o3_name

                        o4_name = '%s.png' % (image_name + '_attn_' + str(epoch))
                        o4_path = image_path + '/' + o4_name

                        save_images(pred, u_path)

            if epoch == 80 or epoch == 120 and total_step != 0:
                # logging.info('train %03d %f', epoch, loss)
                model.eval()
                with torch.no_grad():
                    for _, (input, gt, image_name) in enumerate(test_loader_full):
                        input = Variable(input, volatile=True).cuda()
                        # gt = Variable(gt, volatile=True).cuda()

                        image_name = image_name[0].split('/')[-1].split('.')[0]

                        pred = model(input)
                

                        u_name = '%s.png' % (image_name)
                        u_path = image_path_full_video + '/' + u_name
                        # saving intermediate outputs
                        o1_name = '%s.png' % (image_name + '_GT_' + str(epoch))
                        o1_path = image_path_full_video + '/' + o1_name

                        o3_name = '%s.png' % (image_name + '_gt_' + str(epoch))
                        o3_path = image_path + '/' + o3_name

                        o4_name = '%s.png' % (image_name + '_attn_' + str(epoch))
                        o4_path = image_path + '/' + o4_name

                        save_images(pred, u_path)
         

end = time.time() 
logging.info("end time = %s", end)

run_time = end-start


logging.info("run-time = %s", run_time)

