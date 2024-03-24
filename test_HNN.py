import os 
import numpy as np 
import argparse 
import torch.utils 
import torch.backends.cudnn as cudnn 
from PIL import Image 
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as PSNR 
from skimage.metrics import structural_similarity as SSIM 
from dataloader_NTIRE import NTIRELoader
from dataloader_SR import SrHRNLoader
from resblock_256 import conv_relu_res_relu_block, resblock

device = torch.device("cuda")

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--version", help='experiments version')
parser.add_argument("-ct", "--continue_train", help="resume training if abruptly stopped", default=0)
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
parser.add_argument("-o", "--save", type=str, default='EXP', help='outputs path')
parser.add_argument('--accelerator', type=str, default='gpu', help='gpu or cpu', choices=['gpu', 'cpu', 'tpu'])
parser.add_argument('--trainer', type=str, default='TLIT', choices=['TLIT', 'PT', 'TLITAMP'], help='trainer setting')
parser.add_argument('--dataset', type=str, default='SIDD', choices=['GoPro', 'HIDE_dataset', 'SIDD', 'RealBlur', 'reds', 'redsvtsr','stblur', 'HazeRemoval', 'night-photography-rendering', 'lowlight_track', 'ShadowRemovalT1', 'Mix', 'FLICKR'],help='dataset to train on')
parser.add_argument('--machine', type=str, default='pegasus', choices=['pegasus', 'viserion', 'dgx'],help='machine to train on')
parser.add_argument('--task', type=str, default='denoising', choices=['denoising', 'deblurring', 'dehazing', 'NTIRE24_Challenges', 'SR'])
parser.add_argument('--scale_factor', type=int, help='specify scaling factor for SR task')
parser.add_argument('--crop_size', type = int, default = 256, help = 'crop size')
parser.add_argument('--precision', type = str, default=None, help='choose precision mode for training')
parser.add_argument('--crop', action='store_true')





args = parser.parse_args()


def save_images(tensor, path):
    # image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(tensor.cpu().float().numpy(), (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def compute_metrics(self, input, gt):

    input = input[0].cpu().float().numpy()
    input = (np.transpose(input, (1, 2, 0)))
    gt = gt[0].cpu().float().numpy()
    gt = (np.transpose(gt, (1, 2, 0)))

    # print(input.shape, gt.shape)

    psnr = PSNR(gt, input)
    ssim = SSIM(gt, input, multichannel=True)
    
    return psnr, ssim

test_data = NTIRELoader(img_dir='/home/pegasus/DATA/Nikhil/LLIE/NTIRE24_Challenges/ShadowRemovalT1/results',
                        opt=args)

# test_data = SrHRNLoader(img_dir="/home/pegasus/DATA/Nikhil/LLIE/NTIRE24_Challenges/lowlight_track/testing",
#                                 opt=args)

model = resblock(args, conv_relu_res_relu_block, 16, 3, 3)
# model.load_state_dict(torch.load('/home/pegasus/Experiments/Nikhil/HNN_codebase/DGX_weights/weights_10.pt', map_location=device))
model = model.to(device)


state_dict = torch.load('/home/pegasus/Experiments/Nikhil/HNN_codebase/SHDWREM/shadowremovalt1/Train-20240226-144512/model_epochs/weights_890.pt', map_location=device)
new_state_dict = {}
for key in state_dict:
    new_key = key.replace('module.','')
    new_state_dict[new_key] = state_dict[key]

model.load_state_dict(new_state_dict)
model.eval()

with torch.no_grad():
    for _, (input, gt, image_name) in enumerate(test_data):
        input = Variable(input, volatile=True).cuda()
        # input = input.to(device)
        # gt = Variable(gt, volatile=True).cuda()
        image_name = image_name.split('/')[-1].split('.')[0]
        print(image_name)

        pred = model(input)
        print(pred.shape)


        u_name = '%s.png' % (image_name)
        u_path = '/home/pegasus/Experiments/Nikhil/HNN_codebase/NTIRE24_Challenge_results/shadowt1/' + u_name
        # saving intermediate outputs
        o1_name = '%s.png' % (image_name + '_GT_' )
        o1_path = '/home/pegasus/Experiments/Nikhil/HNN_codebase/NTIRE24_Challenge_results/shadowt1/' + o1_name

        o3_name = '%s.png' % (image_name + '_gt_' )
        o3_path = '/home/pegasus/Experiments/Nikhil/HNN_codebase/NTIRE24_Challenge_results/shadowt1/' + o3_name


        save_images(pred, u_path)
        # save_images(gt, o1_path)
        # save_images(input, o3_path)
        print("saved")

