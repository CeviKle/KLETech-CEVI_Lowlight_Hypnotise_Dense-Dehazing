# import torch
# import torch.nn as nn
# from math import sqrt

# def conv3x3(in_channels, out_channels):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
#                      stride=1, padding=1, bias=True)


# def upsample(scale_factor):
#     return nn.Upsample(scale_factor=scale_factor, mode='bilinear')

# class UpSampleBlock(nn.Module):
#     def __init__(self, in_ch, scale_factor):
#         super().__init__()
#         self.upsample = nn.Upsample(scale_factor=scale_factor,mode='nearest')
#         self.conv = conv3x3(in_ch, in_ch)
#         self.acti = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.acti(self.conv(self.upsample(x)))

# class conv_relu_res_relu_block(nn.Module):
#     def __init__(self):
#         super(conv_relu_res_relu_block, self).__init__()
#         self.conv1 = conv3x3(256, 256)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(256, 256)
#         self.relu2 = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.relu1(out)
#         out = self.conv2(out)
#         out = torch.add(out,residual) 
#         out = self.relu2(out)
#         return out

    
# class resblock(nn.Module):
#     def __init__(self, 
#                  opt, 
#                  block, 
#                  block_num, 
#                  input_channel, 
#                  output_channel):
#         super(resblock, self).__init__()

#         self.opt = opt
#         self.in_channels = input_channel
#         self.out_channels = output_channel
#         self.input_conv = conv3x3(self.in_channels, out_channels=256)  
#         self.conv_seq = self.make_layer(block, block_num)
#         self.conv = conv3x3(256, 256)
#         self.relu = nn.ReLU(inplace=True)
#         self.output_conv = conv3x3(in_channels=256,  out_channels=self.out_channels)
#         # self.upsl = upsample(scale_factor=self.opt.scale_factor)
#         self.upsl = UpSampleBlock(in_ch=256, scale_factor=self.opt.scale_factor)
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
#                 m.weight.data.normal_(0,sqrt(2./n))# the devide  2./n  carefully  
                
#     def make_layer(self,block,num_layers):
#         layers = []
#         for i in range(num_layers):
#             layers.append(block()) # there is a () 
#         return nn.Sequential(*layers)   
    
#     def forward(self, x):
       
#         out = self.input_conv(x)
#         residual = out
#         out = self.conv_seq(out)
#         out = self.conv(out)
#         out = torch.add(out,residual)  
#         out = self.relu(out)
        
#         if self.opt.task == 'SR':
#             # out = UpSampleBlock(out, self.opt.scale_factor)
#             out = self.upsl(out)
#             out = self.output_conv(out)
#         else:
#             out = self.output_conv(out)
#         return out



# dummy
import torch
import torch.nn as nn
from math import sqrt

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=True)


def upsample(scale_factor):
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear')

class UpSampleBlock(nn.Module):
    def __init__(self, in_ch, scale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor,mode='nearest')
        self.conv = conv3x3(in_ch, in_ch)
        self.acti = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.acti(self.conv(self.upsample(x)))

class conv_relu_res_relu_block(nn.Module):
    def __init__(self):
        super(conv_relu_res_relu_block, self).__init__()
        self.conv1 = conv3x3(256, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(256, 256)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = torch.add(out,residual) 
        out = self.relu2(out)
        return out

    
class resblock(nn.Module):
    def __init__(self, 
                 opt, 
                 block, 
                 block_num, 
                 input_channel, 
                 output_channel):
        super(resblock, self).__init__()

        self.opt = opt
        self.in_channels = input_channel
        self.out_channels = output_channel
        self.input_conv = conv3x3(self.in_channels, out_channels=256)  
        self.conv_seq = self.make_layer(block, block_num)
        self.conv = conv3x3(256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.output_conv = conv3x3(in_channels=256,  out_channels=self.out_channels)
        self.upsl = upsample(scale_factor=self.opt.scale_factor)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,sqrt(2./n))# the devide  2./n  carefully  
                
    def make_layer(self,block,num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(block()) # there is a () 
        return nn.Sequential(*layers)   
    
    def forward(self, x):
       
        out = self.input_conv(x)
        residual = out
        out = self.conv_seq(out)
        out = self.conv(out)
        out = torch.add(out,residual)  
        out = self.relu(out)
        
        if self.opt.task == 'SR':
            out = UpSampleBlock(out)
        else:
            out = self.output_conv(out)
        return out



