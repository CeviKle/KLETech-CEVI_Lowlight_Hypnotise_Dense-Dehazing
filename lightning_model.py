import lightning as L 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from vainF_ssim import MS_SSIM
from loss_utils import Vgg16
from PIL import Image 
import numpy as np 
import loss_utils
import os 
from skimage.metrics import peak_signal_noise_ratio as PSNR 
# from skimage.metrics import structural_similarity as SSIM 
from torchmetrics.image import StructuralSimilarityIndexMeasure
from loss_utils import FocalFrequencyLoss as FFL



device = torch.device("cuda")
l1_loss = torch.nn.L1Loss()
feature_loss = Vgg16().to(device)
fftloss = FFL(loss_weight=1.0, alpha=1.0)

dssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


# Learning rate decrease
def adjust_learning_rate(opt, epoch, iteration, optimizer):
    # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
    if opt.lr_decrease_mode == 'epoch':
        lr = opt.lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if opt.lr_decrease_mode == 'iter':
        lr = opt.lr * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class LightningModel(L.LightningModule):
    def __init__(self, generator, opt):
        super().__init__()
        self.generator = generator 
        self.opt = opt    
        # self.save_hyperparameters()

    def compute_metrics(self, input, gt):

        input_np = input[0].cpu().float().numpy()
        input_np = (np.transpose(input_np, (1, 2, 0)))
        gt_np = gt[0].cpu().float().numpy()
        gt_np = (np.transpose(gt_np, (1, 2, 0)))

        # print(input.shape, gt.shape)

        psnr = PSNR(gt_np, input_np)
        ssim = StructuralSimilarityIndexMeasure(data_range=255.0).cuda()
        ssim_val = ssim(input, gt)
        # ssim_val = None

        
        return psnr, ssim_val

    def training_step(self, batch, batch_idx):
        (inp, gt, _) = batch
        pred = self.generator(inp)
        # print(self.current_epoch)

        if self.opt.precision == 16: 
            gt = gt.type('torch.cuda.HalfTensor')
        
        loss_l1 = l1_loss(pred, gt)
        loss_fl = feature_loss(pred, gt, which='relu2')
        loss_ssim = 1-dssim(pred, gt)


        loss = self.opt.alpha*loss_l1 + self.opt.beta * loss_fl + self.opt.gamma * loss_ssim
        if self.opt.dataset == 'HazeRemoval':
            lossfft = fftloss(pred, gt)
            # losscolor = color_loss(pred, gt)
            loss = self.opt.alpha*loss_l1 + self.opt.beta * loss_fl + self.opt.gamma * lossfft + self.opt.gamma * loss_ssim


        if self.current_epoch % 10 == 0:
            model_path = self.opt.save + '/model_epochs/'
            loss_utils.save(self.generator, os.path.join(model_path, 'weights_%d.pt' % self.current_epoch))
        
        return loss 
    
    
    def validation_step(self, batch, epoch):
        (input, gt, image_name) = batch 
        pred = self.generator(input)

        if self.opt.precision == 16: 
            gt = gt.type('torch.cuda.HalfTensor')

        
        image_path = self.opt.save + '/image_epochs/'
        image_name = image_name[0].split('/')[-1].split('.')[0]
        u_name = '%s.png' % (image_name + '_restored_' + str(self.current_epoch))
        
        u_path = image_path + '/' + u_name
        # saving intermediate outputs
        o1_name = '%s.png' % (image_name + '_GT_' + str(self.current_epoch))
        o1_path = image_path + '/' + o1_name

        o3_name = '%s.png' % (image_name + '_input_' + str(self.current_epoch))
        o3_path = image_path + '/' + o3_name

        o4_name = '%s.png' % (image_name + '_attn_' + str(self.current_epoch))
        o4_path = image_path + '/' + o4_name

        save_images(pred, u_path)
        save_images(gt, o1_path)
        save_images(input, o3_path)

        psnr, ssim_val = self.compute_metrics(pred, gt)

        with open(os.path.join(self.opt.save + '/metric_logs/metrics_log.txt'), 'a') as f:
            f.write(f'EPOCH: {self.current_epoch}, PSNR: {psnr}, SSIM: {ssim_val}\n')



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.opt.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [lr_scheduler]
        return optimizer