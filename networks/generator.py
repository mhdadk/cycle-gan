import torch

import torch.nn as nn

from collections import OrderedDict

class CycleGANGenerator(nn.Module):
    
    def __init__(self,num_input_channels,num_residual_blocks):
        
        super(CycleGANGenerator,self).__init__()
        
        # generator parameters
        
        self.num_input_channels = num_input_channels
        
        self.num_residual_blocks = num_residual_blocks
        
        # initial convolutional block (c7s1 in the paper)
        
        self.in_block = nn.Sequential(
        
            nn.ReflectionPad2d(padding = self.num_input_channels),
            
            nn.Conv2d(in_channels = self.num_input_channels,
                      out_channels = 64,
                      kernel_size = (7,7),
                      stride = (1,1),
                      padding = 0,
                      bias = True),
            
            nn.InstanceNorm2d(num_features = 64),
            
            nn.ReLU(inplace = True)            
        
        )
        
        # down-sampling blocks
        
        self.DS1 = self.DS_block(in_channels = 64,
                                 out_channels = 128,
                                 kernel_size = (3,3),
                                 pad_size = 1,
                                 stride = (2,2))
        
        self.DS2 = self.DS_block(in_channels = 128,
                                 out_channels = 256,
                                 kernel_size = (3,3),
                                 pad_size = 1,
                                 stride = (2,2))
        
        # residual block forward paths
        
        self.res_forward = self.residual_forward(in_channels = 256,
                                                 out_channels = 256,
                                                 kernel_size = (3,3),
                                                 pad_size = 1,
                                                 stride = (1,1))
        
        # up-sampling blocks
        
        self.US1 = self.US_block(scale_factor = 2,
                                 in_channels = 256,
                                 out_channels = 128,
                                 kernel_size = (3,3),
                                 pad_size = 1,
                                 stride = (1,1))
        
        self.US2 = self.US_block(scale_factor = 2,
                                 in_channels = 128,
                                 out_channels = 64,
                                 kernel_size = (3,3),
                                 pad_size = 1,
                                 stride = (1,1))
        
        # final output block
        
        self.out_block = nn.Sequential(
            
            nn.ReflectionPad2d(padding = self.num_input_channels),
            
            nn.Conv2d(in_channels = 64,
                      out_channels = self.num_input_channels,
                      kernel_size = (7,7),
                      stride = (1,1),
                      padding = 0,
                      bias = True),
            
            nn.Tanh()
        
        )

    def residual_forward(self,in_channels,out_channels,kernel_size,pad_size,
                         stride):
    
        layers = OrderedDict()
        
        layers['pad1'] = nn.ReflectionPad2d(padding = pad_size)
        
        layers['conv1'] = nn.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = 0,
                                    bias = True)
        
        layers['IN1'] = nn.InstanceNorm2d(num_features = out_channels)
        
        layers['relu1'] = nn.ReLU(inplace = True)
        
        layers['pad2'] = nn.ReflectionPad2d(padding = pad_size)
        
        layers['conv2'] = nn.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = 0,
                                    bias = True)
        
        layers['IN2'] = nn.InstanceNorm2d(num_features = out_channels)
                
        return nn.Sequential(layers)

    def DS_block(self,in_channels,out_channels,kernel_size,pad_size,
                 stride):
        
        layers = OrderedDict()
        
        layers['conv1'] = nn.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = pad_size,
                                    bias = True)
        
        layers['IN1'] = nn.InstanceNorm2d(num_features = out_channels)
        
        layers['relu1'] = nn.ReLU(inplace = True)
            
        return nn.Sequential(layers)

    def US_block(self,scale_factor,in_channels,out_channels,kernel_size,
                 pad_size,stride):
        
        layers = OrderedDict()
        
        layers['US1'] = nn.Upsample(scale_factor = scale_factor)
        
        layers['conv1'] = nn.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = pad_size,
                                    bias = True)
        
        layers['IN1'] = nn.InstanceNorm2d(num_features = out_channels)
        
        layers['relu1'] = nn.ReLU(inplace = True)
        
        return nn.Sequential(layers)
    
    def forward(self,x):
        
        # initial convolutional block
        
        x = self.in_block(x) # 1 conv layer
        
        # down-sampling blocks
        
        x = self.DS1(x) # 1 conv layer
        x = self.DS2(x) # 1 conv layer
        
        # residual blocks
        
        for _ in range(self.num_residual_blocks):
            
            # add residual to forward path
            
            x = self.res_forward(x) + x # 2 x 9 = 18 conv layers
        
        # up-sampling blocks
        
        x = self.US1(x) # 1 conv layer
        x = self.US2(x) # 1 conv layer
        
        # output block
        
        x = self.out_block(x) # 1 conv layer
        
        return x

# test

if __name__ == '__main__':
    
    """
    
    nn.Conv2d expects an input shape of (N,in_channels,H,W), where:
        
        N = batch size
        in_channels = number of input channels
        H = image height (number of rows)
        W = image width (number of columns)
    
    in this case, for input x, N = 8, in_channels = 128, H = 256, W = 256
    
    NOTE: a batch of 8 256 x 256 images requires too much memory, will throw
    error. Use a batch size of 1 instead.
    
    NOTE: Image dimensions must be even so that input_dim = output_dim
    
    """
    
    x = torch.randn((8,3,64,64))
    
    generator = CycleGANGenerator(x.shape[1],9)
    
    y = generator(x)
    
    show_images = False
    
    if show_images:
        
        import torchvision.utils
        import matplotlib.pyplot as plt
        import numpy as np
        
        imgs = torch.cat((x,y))
        
        img_grid = torchvision.utils.make_grid(imgs,nrow=4,padding=10)
        
        def show(img):
            npimg = img.detach().numpy()
            plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
            
        show(img_grid)