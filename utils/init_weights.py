from torch import nn

# function used to initialize weights of convolutional layers

def init_weights(layer):
    
    layer_name = layer.__class__.__name__
        
    if 'Conv' in layer_name:
        
        # this is done in-place
        
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        
        if hasattr(layer, 'bias') and layer.bias is not None:

            # this is also done in-place
    
            nn.init.constant_(layer.bias.data, 0.0)

# test

if __name__ == '__main__':

    from cyclegan_gen import CycleGANGenerator
    
    import torch
    
    x = torch.randn((1,3,256,256))
    
    generator = CycleGANGenerator(x.shape[1],9)
    
    generator.apply(init_weights)