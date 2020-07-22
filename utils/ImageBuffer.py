import torch

import random

class ImageBuffer():
    
    def __init__(self,buffer_size):
        
        self.buffer_size = buffer_size
        
        self.buffer = []
                
    def get_batch(self, in_batch):
        
        out_batch = []
                
        # add another dimension for concatenation later. The resulting tensor
        # will be of shape (batch_size,1,num_channels,img_height,img_width)
        
        in_batch = torch.unsqueeze(in_batch,1)
        
        for image in in_batch: # iterate along first axis
            
            # if the buffer is not yet full
            
            if len(self.buffer) < self.buffer_size:
                
                self.buffer.append(image)
                out_batch.append(image)
            
            else:
                
                """
            
                half of the output batch will be made up of previously generated
                images, while the other half will be made up of half of the images
                from the input batch, which was just generated. See the "training
                details" under section 4 in the original CycleGAN paper and also
                see section 2.3 in "Learning from Simulated and Unsupervised Images
                through Adversarial Training":
                
                https://arxiv.org/pdf/1612.07828.pdf
                        
                """
                
                p = random.uniform(0,1)
                
                # TODO: try using from_buffer = 1 then 1 - from_buffer instead
                # of this
                
                if p > 0.5:
                
                    # randint is inclusive
                
                    rand_idx = random.randint(0,self.buffer_size - 1)
                    
                    # sample from buffer
                    
                    out_batch.append(self.buffer[rand_idx].clone())
                
                    # replace (batch_size/2) samples in the buffer with the newly
                    # generated images
                
                    self.buffer[rand_idx] = image
            
                else:
                    
                    out_batch.append(image)
    
        # compile the images into a batch
        
        out_batch = torch.cat(out_batch,0)
        
        return out_batch

# test

if __name__ == '__main__':
    
    buffer_size = 50
    
    img_buffer = ImageBuffer(buffer_size)
        
    y = []
    
    for _ in range(100):
        
        x = torch.randn((8,3,32,32))
        
        y.append(img_buffer.get_batch(x))
    
    