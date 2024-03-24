import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import models
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import normalized_root_mse as NRMSE
import rawpy
import glob
import imageio
import cv2
import torchvision
import os
import numpy as np
import shutil
from torch.nn.modules.container import T
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn


def define_weights(num):
    weights = np.float32((np.logspace(0,num,127, endpoint=True, base=10.0)))
    weights = weights/np.max(weights)
    weights = np.flipud(weights).copy()    
    return weights

def get_na(bins,weights,img_loww,amp=5):
    H,W = img_loww.shape
    arr = img_loww*1
    selection_dict = {weights[0]: (bins[0]<=arr)&(arr<bins[1])}
    for ii in range(1,len(weights)):
        selection_dict[weights[ii]] = (bins[ii]<=arr)&(arr<bins[ii+1])
    mask = np.select(condlist=selection_dict.values(), choicelist=selection_dict.keys())
   
    mask_sum1 = np.sum(mask,dtype=np.float64)
    
    na1 = np.float32(np.float64(mask_sum1*0.01*amp)/np.sum(img_loww*mask,dtype=np.float64))
# As in SID max amplification is limited to 300
    if na1>300.0:
        na1 = np.float32(300.0)
    if na1<1.0:
        na1 = np.float32(1.0)
    
    selection_dict.clear()

    return na1


def part_init(gt_files,train_files,num_print,filename,gt_amp=False):

    file_line = open(filename, 'w')
    bins = np.float32((np.logspace(0,8,128, endpoint=True, base=2.0)-1))/255.0
    print('\nEdges:{}, dtype:{}\n'.format(bins,bins.dtype), file = file_line)
    weights5 = define_weights(5)
    print('------- weights: {}\n'.format(weights5), file = file_line)

    gt_list = []
    train_list = []
    mean = 0
    
    for i in range(len(gt_files)):
        raw = rawpy.imread(gt_files[i])
        img_gt = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16).copy()
        raw.close()            
        img_gtt=np.float32(img_gt/65535.0)
        h,w,_ = img_gtt.shape
        
        correct_dim_flag = False
        if h%32!=0:
            print('Correcting the 1st dimension.')
            h = (h//32)*32
            img_gtt = img_gtt[:h,:,:]
            correct_dim_flag = True
        
        if w%32!=0:
            print('Correcting the 2nd dimension.')
            w = (w//32)*32
            img_gtt = img_gtt[:,:w,:]
            correct_dim_flag = True
            
        gt_list.append(img_gtt)
        
        raw = rawpy.imread(train_files[i])
        img = raw.raw_image_visible.astype(np.float32).copy()
        raw.close()
        
        if correct_dim_flag:
            img = img[:h,:w]        
        
        img_loww = (np.maximum(img - 512,0)/ (16383 - 512))       
        
        na5 = get_na(bins,weights5,img_loww)
        
        if gt_files[i][-7]=='3':
            ta=300
        else:
            ta=100
        
        H,W = img_loww.shape    
        a = np.float32(np.float64(H*W*0.01)/np.sum(img_loww,dtype=np.float64))
        
        if gt_amp:
            img_loww = (img_loww*ta)
            print('...using gt_amp : {}'.format(gt_files[i][-17:]), file = file_line)
        else:
            img_loww = (img_loww*na5)
            print('...using na5 : {}'.format(gt_files[i][-17:]), file = file_line)
            
        train_list.append(img_loww)
        mean += np.mean(img_loww[0::2,1::2],dtype=np.float32)

        if (i+1)%num_print==0:
            print('... files loading : {}'.format(gt_files[i][-17:]))
            print('Image {} base_amp: {}, gt_amp: {}, Our_Amp:{}'.format(i+1,a,ta,na5))
        
        print('Image {} base_amp: {}, gt_amp: {}, Our_Amp:{}'.format(i+1,a,ta,na5), file = file_line)
   
    print('Files loaded : {}/{}, channel mean: {}'.format(len(train_list), len(gt_files), mean/len(train_list)))
    file_line.close()
    return gt_list, train_list
    
    
################ DATASET CLASS
class load_data(Dataset):
    """Loads the Data."""
    
    def __init__(self, train_files, gt_files, filename, num_print, gt_amp=True, training=True):        
        
        self.training = training
        if self.training:
            print('\n...... Train files loading\n')
            self.gt_list, self.train_list = part_init(gt_files,train_files,num_print,filename,gt_amp)        
            print('\nTrain files loaded ......\n')
        else:
            print('\n...... Test files loading\n')
            self.gt_list, self.train_list = part_init(gt_files,train_files,num_print,filename,gt_amp)        
            print('\nTest files loaded ......\n')
        
    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
    
        img_gtt = self.gt_list[idx]
        img_loww = self.train_list[idx]
        
        H,W = img_loww.shape
        
        if self.training:
            i = random.randint(0, (H-512-2)//2)*2
            j = random.randint(0,(W-512-2)//2)*2

            img_low = img_loww[i:i+512,j:j+512]
            img_gt = img_gtt[i:i+512,j:j+512,:]
            
            if random.randint(0, 100)>50:
                img_gt = np.fliplr(img_gt).copy()
                img_low = np.fliplr(img_low).copy()

            if random.randint(0, 100)<20:
                img_gt = np.flipud(img_gt).copy()
                img_low = np.flipud(img_low).copy()
        else:
            img_low = img_loww
            img_gt = img_gtt
            
        gt = torch.from_numpy((np.transpose(img_gt, [2, 0, 1]))).float()
        low = torch.from_numpy(img_low).float().unsqueeze(0)            
        
        return low, gt
        

        
#################### perceptual loss

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
        self.l1 = torch.nn.L1Loss()
        
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x].eval())
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x].eval())
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x].eval())
#        for x in range(16, 23):
#            self.slice4.add_module(str(x), vgg_pretrained_features[x].eval())
        for name, param in self.named_parameters():
            param.requires_grad = False
            print(name,' grad of VGG set to false !!')
    
    def VGGfeatures(self, x):
        
        x = self.slice1(x)
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
#            h = self.slice4(h)
#            h_relu4_3 = h
            
        return relu2_2, relu3_3


    def forward(self, ip, target, which='relu2'):
        
        ip = (ip-self.mean) / self.std
        target = (target-self.mean) / self.std
    
        ip_relu2_2, ip_relu3_3 = self.VGGfeatures(ip)
        target_relu2_2, target_relu3_3 = self.VGGfeatures(target)
    
        if which=='relu2':
            loss = self.l1(ip_relu2_2,target_relu2_2)
        elif which=='relu3':
            loss = self.l1(ip_relu3_3,target_relu3_3)
        elif which=='both':
            loss = self.l1(ip_relu2_2,target_relu2_2) + self.l1(ip_relu3_3,target_relu3_3)
        else:
            raise NotImplementedError('Incorrect WHICH in perceptual loss.')
        
        return loss
        

    
######### testing


def run_test(model, dataloader_test, iteration, save_images, save_csv_files, metric_average_filename, mode, training=True):
    psnr = ['PSNR']
    ssim = ['SSIM']
    C_pred = ['Colorfulness_pred']
    C_gt = ['Colorfulness_gt']
    
    with torch.no_grad():
        model.eval()
        for image_num, img in enumerate(dataloader_test):
            low = img[0].to(next(model.parameters()).device)
            gt = img[1]
            low = torch.permute(low, (0,3,1,2))
            gt = torch.permute(gt, (0,3,1,2))

            pred = model(low)
            
            pred = (np.clip(pred[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
            gt = (np.clip(gt[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
            psnr_img = PSNR(pred,gt)
            ssim_img = SSIM(pred,gt,multichannel=True)
            c_gt = 0
            c_pred = 0
            
            cond = True
            if training:
                cond = image_num in [0,1,2,3,7,10,11,12,13,19,20,30,35,41,46,47,48] # During training testing will be done only for few test images. Include those in this list.
            
            if cond:
                # imageio.imwrite(os.path.join(save_images,'{}_{}_gt_C_{}.jpg'.format(image_num,iteration,c_gt)), gt)
                # imageio.imwrite(os.path.join(save_images,'{}_{}_psnr_{}_ssim_{}_C_{}.jpg'.format(image_num,iteration, psnr_img, ssim_img,c_pred)), pred)
                cv2.imwrite(os.path.join(save_images,'{}_{}_gt_C_{}.jpg'.format(image_num,iteration,c_gt)), gt)
                cv2.imwrite(os.path.join(save_images,'{}_{}_psnr_{}_ssim_{}_C_{}.jpg'.format(image_num,iteration, psnr_img, ssim_img,c_pred)), pred)
            
            psnr.append(psnr_img)
            ssim.append(ssim_img)
            C_pred.append(c_pred)
            C_gt.append(c_gt)
            
    np.savetxt(os.path.join(save_csv_files,'Metrics_iter_{}.csv'.format(iteration)), [p for p in zip(psnr,ssim,C_pred,C_gt)], delimiter=',', fmt='%s')
    
    psnr_avg = sum(psnr[1:]) / len(psnr[1:])
    ssim_avg = sum(ssim[1:]) / len(ssim[1:])
    c_gt_avg = sum(C_gt[1:]) / len(C_gt[1:])
    c_pred_avg = sum(C_pred[1:]) / len(C_pred[1:])

    f =  open(metric_average_filename, mode)
    f.write('-- psnr_avg:{}, ssim_avg:{}, c_gt_avg:{}, c_pred_avg:{}, iter:{}\n'.format(psnr_avg,ssim_avg,c_gt_avg,c_pred_avg,iteration))
    print('metric average printed.')        
    f.close()

    return
    

class LuminanceLoss(torch.nn.Module):
    """ 
    This loss function calculates the luminance drop 
    between predicted image and the ground truth
    """
    def __init__(self):
        super(LuminanceLoss, self).__init__()
        
    def forward(self, pred, gt):
        self.x = pred
        self.y = gt
        self.x = self.x*255
        self.y = self.y*255
        MuX = torch.mean(self.x)
        MuY = torch.mean(self.y)

        return ((2*MuX*MuY+6.5025)/(MuX**2+MuY**2+6.5025))
            

class ContrastLoss(torch.nn.Module):
    """
    This loss function calculates the contrast drop 
    between the predicted image and the ground truth image
    """
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def forward(self, pred, gt):
        self.a = pred
        self.b = gt
        self.a = self.a*255
        self.b = self.b*255
        SX = torch.std(self.a)
        SY = torch.std(self.b)

        return ((2*SX*SY+58.5225)/(SX**2+SY**2+58.5225))


class Color_Loss(torch.nn.Module):
    def __init__(self):
        super(Color_Loss, self).__init__()
        
    def forward(self, pred, gt):
        self.ground_truth = gt
        self.predicted = pred
        self.ground_truth = self.ground_truth/255
        self.predicted = self.predicted/255
        
        gt_gaussian = torchvision.transforms.functional.gaussian_blur(self.ground_truth, kernel_size=3)
        pred_gaussian = torchvision.transforms.functional.gaussian_blur(self.predicted, kernel_size=3)

        return (torch.sum(((gt_gaussian - pred_gaussian)**2)))
    

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
    # para = 0.0
    # for name, v in model.named_parameters():
    #     if v.requires_grad == True:
    #         if "auxiliary" not in name:
    #             para += np.prod(v.size())
    # return para / 1e6
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6



def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path,exist_ok=True)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'),exist_ok=True)
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def rgb_to_ycbcr(input: torch.Tensor, consts='yuv'):
    return rgb_to_yuv(input, consts == 'ycbcr')

def rgb_to_yuv(input: torch.Tensor, consts='yuv'):
    """Converts one or more images from RGB to YUV.
    Outputs a tensor of the same shape as the `input` image tensor, containing the YUV
    value of the pixels.
    The output is only well defined if the value in images are in [0,1].
    Y′CbCr is often confused with the YUV color space, and typically the terms YCbCr
    and YUV are used interchangeably, leading to some confusion. The main difference
    is that YUV is analog and YCbCr is digital: https://en.wikipedia.org/wiki/YCbCr
    Args:
      input: 2-D or higher rank. Image data to convert. Last dimension must be
        size 3. (Could add additional channels, ie, AlphaRGB = AlphaYUV)
      consts: YUV constant parameters to use. BT.601 or BT.709. Could add YCbCr
        https://en.wikipedia.org/wiki/YUV
    Returns:
      images: images tensor with the same shape as `input`.
    """

    #channels = input.shape[0]

    if consts == 'BT.709': # HDTV YUV
        Wr = 0.2126
        Wb = 0.0722
        Wg = 1 - Wr - Wb #0.7152
        Uc = 0.539
        Vc = 0.635
        delta: float = 0.5 #128 if image range in [0,255]
    elif consts == 'ycbcr': # Alt. BT.601 from Kornia YCbCr values, from JPEG conversion
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
        Uc = 0.564 #(b-y) #cb
        Vc = 0.713 #(r-y) #cr
        delta: float = .5 #128 if image range in [0,255]
    elif consts == 'yuvK': # Alt. yuv from Kornia YUV values: https://github.com/kornia/kornia/blob/master/kornia/color/yuv.py
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
        Ur = -0.147
        Ug = -0.289
        Ub = 0.436
        Vr = 0.615
        Vg = -0.515
        Vb = -0.100
        #delta: float = 0.0
    elif consts == 'y': #returns only Y channel, same as rgb_to_grayscale()
        #Note: torchvision uses ITU-R 601-2: Wr = 0.2989, Wg = 0.5870, Wb = 0.1140
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
    else: # Default to 'BT.601', SDTV YUV
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
        Uc = 0.493 #0.492
        Vc = 0.877
        delta: float = 0.5 #128 if image range in [0,255]

    r: torch.Tensor = input[..., 0, :, :]
    g: torch.Tensor = input[..., 1, :, :]
    b: torch.Tensor = input[..., 2, :, :]
    #TODO
    #r, g, b = torch.chunk(input, chunks=3, dim=-3) #Alt. Which one is faster? Appear to be the same. Differentiable? Kornia uses both in different places

    if consts == 'y':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        #(0.2989 * input[0] + 0.5870 * input[1] + 0.1140 * input[2]).to(img.dtype)
        return y
    elif consts == 'yuvK':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        u: torch.Tensor = Ur * r + Ug * g + Ub * b
        v: torch.Tensor = Vr * r + Vg * g + Vb * b
    else: #if consts == 'ycbcr' or consts == 'yuv' or consts == 'BT.709':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        u: torch.Tensor = (b - y) * Uc + delta #cb
        v: torch.Tensor = (r - y) * Vc + delta #cr

    if consts == 'uv': #returns only UV channels
        return torch.stack((u, v), -3)
    else:
        return torch.stack((y, u, v), -3)

def ycbcr_to_rgb(input: torch.Tensor):
    return yuv_to_rgb(input, consts = 'ycbcr')

def yuv_to_rgb(input: torch.Tensor, consts='yuv') -> torch.Tensor:
    if consts == 'yuvK': # Alt. yuv from Kornia YUV values: https://github.com/kornia/kornia/blob/master/kornia/color/yuv.py
        Wr = 1.14 #1.402
        Wb = 2.029 #1.772
        Wgu = 0.396 #.344136
        Wgv = 0.581 #.714136
        delta: float = 0.0
    elif consts == 'yuv' or consts == 'ycbcr': # BT.601 from Kornia YCbCr values, from JPEG conversion
        Wr = 1.403 #1.402
        Wb = 1.773 #1.772
        Wgu = .344 #.344136
        Wgv = .714 #.714136
        delta: float = .5 #128 if image range in [0,255]

    #Note: https://github.com/R08UST/Color_Conversion_pytorch/blob/75150c5fbfb283ae3adb85c565aab729105bbb66/differentiable_color_conversion/basic_op.py#L65 has u and v flipped
    y: torch.Tensor = input[..., 0, :, :]
    u: torch.Tensor = input[..., 1, :, :] #cb
    v: torch.Tensor = input[..., 2, :, :] #cr
    #TODO
    #y, u, v = torch.chunk(input, chunks=3, dim=-3) #Alt. Which one is faster? Appear to be the same. Differentiable? Kornia uses both in different places

    u_shifted: torch.Tensor = u - delta #cb
    v_shifted: torch.Tensor = v - delta #cr

    r: torch.Tensor = y + Wr * v_shifted
    g: torch.Tensor = y - Wgv * v_shifted - Wgu * u_shifted
    b: torch.Tensor = y + Wb * u_shifted
    return torch.stack((r, g, b), -3)

#Not tested:
def rgb2srgb(imgs):
    return torch.where(imgs<=0.04045,imgs/12.92,torch.pow((imgs+0.055)/1.055,2.4))

#Not tested:
def srgb2rgb(imgs):
    return torch.where(imgs<=0.0031308,imgs*12.92,1.055*torch.pow((imgs),1/2.4)-0.055)




# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight