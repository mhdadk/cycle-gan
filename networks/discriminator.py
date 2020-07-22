import torch

import torch.nn as nn

from collections import OrderedDict

# build a PatchGAN discrminator

class CycleGANDiscriminator(nn.Module):
    
    def __init__(self,num_input_channels):
        
        super(CycleGANDiscriminator,self).__init__()
        
        # discriminator parameters
        
        self.num_input_channels = num_input_channels
        
        # initial down-sampling block
        
        self.in_block = nn.Sequential(
            
            nn.Conv2d(in_channels = self.num_input_channels,
                      out_channels = 64,
                      kernel_size = (4,4),
                      stride = (2,2),
                      padding = 1,
                      bias = True),
            
            nn.LeakyReLU(0.2,inplace = True)
            
        )
        
        # next down-sampling blocks
        
        self.DS1 = self.DS_block(in_channels = 64,
                                 out_channels = 128,
                                 kernel_size = (4,4),
                                 pad_size = 1,
                                 stride = (2,2))
        
        self.DS2 = self.DS_block(in_channels = 128,
                                 out_channels = 256,
                                 kernel_size = (4,4),
                                 pad_size = 1,
                                 stride = (2,2))
        
        # stride = (1,1) leads to non-division by 2, consider padding
        
        self.DS3 = self.DS_block(in_channels = 256,
                                 out_channels = 512,
                                 kernel_size = (4,4),
                                 pad_size = 1,
                                 stride = (1,1))
        
        """
        
        final convolutional layer. Note that this outputs an image where
        every pixel will carry the posterior probability estimate for each
        70 x 70 patch in the original image.
        
        It can classify whether 70×70 overlapping patches are real or fake
        and has fewer parameters than a normal discriminator.
        
        The benefit of using a PatchGAN discriminator is that the loss function
        can then measure how good the discriminator is at distinguishing images
        based on their style rather than their content. Since each individual
        element of the discriminator prediction is based only on a small square
        of the image, it must use the style of the patch, rather than its
        content, to make its decision.
        
        The output image will have a shape of (1,
                                               input_img.shape[0]//8,
                                               input_img.shape[1]//8)
        
        The derivation of the size of the patch being 70 x 70 can be found
        here:
            
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39#issuecomment-336737526
        
        """
        
        self.conv = nn.Conv2d(in_channels = 512,
                              out_channels = 1,
                              kernel_size = (4,4),
                              stride = (1,1),
                              padding = 1,
                              bias = True)
        
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
        
        layers['leakyrelu1'] = nn.LeakyReLU(0.2,inplace = True)
            
        return nn.Sequential(layers)
    
    def forward(self,x):
        
        # initial down-sampling block
        
        x = self.in_block(x)
        
        # next down-sampling blocks
        
        x = self.DS1(x)
        x = self.DS2(x)
        x = self.DS3(x)
        
        # final convolutional layer
                
        x = self.conv(x)
        
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
    
    x = torch.randn((1,3,256,256))
    
    discriminator = CycleGANDiscriminator(x.shape[1])
    
    y = discriminator(x)
        
        
    
        
        