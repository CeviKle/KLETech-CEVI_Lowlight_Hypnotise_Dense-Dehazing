import argparse
import os 
import torch 
import numpy as np 

from dataloader import MSIReconLoader, HRNLoader
from dataloader_gopro import GoProHRNLoader
from dataloader_HIDE import HIDEHRNLoader
from dataloader_RealBlur import RealBlurHRNLoader
from dataloader_NTIRE import NTIRELoader

import utils
import lightning as L 
from lightning_model import LightningModel

parser = argparse.ArgumentParser()


parser.add_argument('--pre_train', type = bool, default = True, help = 'pre_train or not')
parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the testing batches for single GPU')
parser.add_argument('--num_workers', type = int, default = 2, help = 'number of cpu threads to use during batch generation')
parser.add_argument('--val_path', type = str, default = './test', help = 'saving path that is a folder')
parser.add_argument('--task_name', type = str, default = 'track1', help = 'task name for loading networks, saving, and log')
# Network initialization parameters
parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
parser.add_argument('--activ', type = str, default = 'relu', help = 'activation type of networks')
parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
parser.add_argument('--in_channels', type = int, default = 3, help = 'input channels for generator')
parser.add_argument('--out_channels', type = int, default = 3, help = 'output channels for generator')
parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for generator')
parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
# Dataset parameters
parser.add_argument('--baseroot', type = str, default = './NTIRE2020_Test_Clean', help = 'baseroot')
parser.add_argument('--crop_size', type = int, help = 'crop size')
parser.add_argument('--dataset', type=str, default='SIDD', choices=['GoPro', 'HIDE_dataset', 'SIDD', 'RealBlur', 'reds', 'redsvtsr','stblur', 'HazeRemoval', 'night-photography-rendering', 'lowlight_track', 'ShadowRemovalT1'],help='dataset to train on')
parser.add_argument('--machine', type=str, default='pegasus', choices=['pegasus', 'viserion', 'dgx'],help='machine to train on')
parser.add_argument('--task', type=str, default='denoising', choices=['denoising', 'deblurring', 'dehazing', 'NTIRE24_Challenges'])
parser.add_argument('--precision', type = str, default=None, help='choose precision mode for training')

opt = parser.parse_args()

device = torch.device("cuda")

if opt.machine == 'pegasus':
    data_root = f"/home/pegasus/DATA/Nikhil/LLIE/{opt.task }/{opt.dataset}"
if opt.machine == 'viserion':
    data_root = f"/DATA/Nikhil/{opt.task}/{opt.dataset}"
if opt.machine == 'dgx':
    data_root = f"/workspace/data/{opt.task}/{opt.dataset}"
print("Data Located at: ",data_root)

# generator = utils.create_generator_val1(opt, './track1/G_epoch100_bs8.pth').to(device)

if opt.dataset == 'SIDD':
    # LOLDataset
    train_data = HRNLoader(csv_file=f"{data_root}/SIDD.csv",
                        img_dir=f"{data_root}/INPUT/",
                        gt_img_dir=f"{data_root}/GT/",
                    task='train', opt=opt)
    test_data = HRNLoader(csv_file=f"{data_root}/SIDD_test.csv",
                        img_dir=f"{data_root}/INPUT/",
                        gt_img_dir=f"{data_root}/GT/",
                    task='test', opt=opt)

lightningmodel = LightningModel.load_from_checkpoint("/home/pegasus/Experiments/Nikhil/HRN/EXP/sidd_hrn_testing/Train-20240218-153606/model_epochs/weights_30.pt")

trainer = L.Trainer(
    max_epochs=opt.epochs,
    accelerator="gpu",
    # precision="16",
    # strategy="ddp",
    devices=[0],
    # logger=logger,
    deterministic=False,
)

trainer.test(lightningmodel, dataloaders=test_data)



