import torch

import torchvision as tv

import time

import random

import itertools

import time

from PIL import Image

from glob import glob

from networks.generator import CycleGANGenerator

from networks.discriminator import CycleGANDiscriminator

from utils.init_weights import init_weights

from utils.image_buffer import ImageBuffer

from utils.ImageDataset import ImageDataset

def get_dataloader(img_dir,
                   img_transforms,
                   mode = 'train',
                   batch_size = 8,
                   aligned = True,
                   shuffle = True):
    
    dataset = ImageDataset(img_dir = img_dir,
                           transforms = img_transforms,
                           mode = mode,
                           aligned = aligned)
        
    # create iterators for training and testing datasets
        
    dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                             batch_size = batch_size,
                                             shuffle = shuffle,
                                             num_workers = 0)
    
    return dataloader
    
# initialize path to GPU if available
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize models and put them onto GPU if available

G_AB = CycleGANGenerator(num_input_channels = 3,
                         num_residual_blocks = 9).to(device)

G_BA = CycleGANGenerator(num_input_channels = 3,
                         num_residual_blocks = 9).to(device)

D_A = CycleGANDiscriminator(num_input_channels = 3).to(device)

D_B = CycleGANDiscriminator(num_input_channels = 3).to(device)

# initialize model parameters

G_AB.apply(init_weights)
G_BA.apply(init_weights)
D_A.apply(init_weights)
D_B.apply(init_weights)

# initialize generator and discriminator losses See the "training details"
# section under section 4 in the paper for details.

adv_loss_func = torch.nn.MSELoss()

cycle_loss_func = torch.nn.L1Loss()

identity_loss_func = torch.nn.L1Loss()

# initialize optimizers so that both discriminators and generators are
# trained simultaneously

G_optimizer = torch.optim.Adam(
              itertools.chain(G_AB.parameters(),G_BA.parameters()),
              lr = 0.0002,
              betas = (0.5, 0.999))

D_optimizer = torch.optim.Adam(
              itertools.chain(D_A.parameters(),D_B.parameters()),
              lr = 0.0002,
              betas=(0.5, 0.999))

# initialize learning rate schedulers

def lr_decay_factor(epoch_num):
        
    # epoch to start decaying learning rate from
    
    decay_start = 5 # epochs
    
    # epoch to reach learning rate of 0
    
    decay_end = 10 # epochs
    
    # decay factor. This will be equal to 1 until decay_start is reached,
    # after which it will linearly decrease
    
    decay_factor = 1 - (max(0,epoch_num - decay_start) / (decay_end - decay_start))
    
    return decay_factor

G_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = G_optimizer,
                                                lr_lambda = lr_decay_factor)

D_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = D_optimizer,
                                                lr_lambda = lr_decay_factor)

# initialize image dataloaders

img_dir = '../apple2orange'
        
train_img_transforms = tv.transforms.Compose([
                       tv.transforms.RandomHorizontalFlip(),
                       tv.transforms.ToTensor(),
                       tv.transforms.Normalize([0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])])

train_dataloader = get_dataloader(img_dir = img_dir,
                                  img_transforms = train_img_transforms,
                                  mode = 'train',
                                  batch_size = 1,
                                  aligned = False,
                                  shuffle = True)

test_img_transforms = tv.transforms.Compose([
                       tv.transforms.ToTensor(),
                       tv.transforms.Normalize([0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])])

test_dataloader = get_dataloader(img_dir = img_dir,
                                 img_transforms = test_img_transforms,
                                 mode = 'test',
                                 batch_size = 1,
                                 aligned = False,
                                 shuffle = True)

# number of epochs to train for
    
num_epochs = 10

"""

initialize buffer to store 50 previously generated images. See the
"training details" section under section 4 in the paper for details.

"""

buffer_size = 50
buffer_A = ImageBuffer(buffer_size)
buffer_B = ImageBuffer(buffer_size)
    
# track how long training takes

train_start = time.time()

for epoch in range(num_epochs):
    
    # track how long each epoch takes
    
    epoch_start = time.time()
    
    # set models to training mode
    
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()
    
    for img_A,img_B in train_dataloader:
        
        # move images to GPU if available
        
        real_A = img_A.to(device)
        real_B = img_B.to(device)
        
        # start tracking gradients for the images
        
        real_A.requires_grad = True
        real_B.requires_grad = True
        
        # train generators ----------------------------------------------------
        
        # freeze discriminators so that only generators are trained
        
        for param in D_A.parameters():
            param.requires_grad = False
            
        for param in D_B.parameters():
            param.requires_grad = False
        
        # zero generator parameter gradients
        
        G_optimizer.zero_grad()
        
        # adversarial loss ----------------------------------------------------
        
        # B --> A
        
        # G_BA(B)
        
        fake_A = G_BA(real_B)
        
        # the output of D_A is an image that classifies patches of fake_A
        
        D_A_out = D_A(fake_A)
        
        # the generator is trying to make the discriminator classify its
        # generated images as real
        
        D_A_target = torch.ones_like(D_A_out)
        
        G_BA_loss = adv_loss_func(D_A_out,D_A_target)
        
        # A --> B
        
        # G_AB(A)
        
        fake_B = G_AB(real_A)
                
        # the output of D_B is an image that classifies patches of fake_B
        
        D_B_out = D_B(fake_B)
        
        # the generator is trying to make the discriminator classify its
        # generated images as real
        
        D_B_target = torch.ones_like(D_B_out)
        
        G_AB_loss = adv_loss_func(D_B_out,D_B_target)
        
        # total adversarial loss
        
        adv_loss = G_AB_loss + G_BA_loss
        
        # cycle loss ----------------------------------------------------------
        
        # cycle loss weight
        
        lambda_cycle = 10
        
        # G_BA(G_AB(A))
        
        recon_A = G_BA(fake_B)
                
        # compare the reconstructed image to the original image
        
        cycle_lossA = cycle_loss_func(recon_A,real_A)
        
        # G_AB(G_BA(B))
        
        recon_B = G_AB(fake_A)
            
        # compare the reconstructed image to the original image
                
        cycle_lossB = cycle_loss_func(recon_B,real_B)
        
        # total cycle loss
        
        cycle_loss = (cycle_lossA + cycle_lossB) * lambda_cycle
        
        # identity loss -------------------------------------------------------
        
        """
        
        From the section "Photo generation from paintings" under section 5.2
        in the paper, the authors note that it is helpful to introduce an
        additional "identity" loss. This is similar to the loss of an
        autoencoder
        
        """
        
        # G_BA(A)
        
        identity_A = G_BA(real_A)
                    
        identity_lossA = identity_loss_func(identity_A,real_A)
        
        # G_AB(B)
        
        identity_B = G_AB(real_B)
                    
        identity_lossB = identity_loss_func(identity_B,real_B)
        
        identity_loss = (identity_lossA + identity_lossB) * lambda_cycle * 0.5

        # total generator losses
        
        total_G_loss = adv_loss + cycle_loss + identity_loss
        
        # compute gradients with respect to generator parameters only since
        # discriminator parameters are frozen
        
        total_G_loss.backward()
        
        # update parameters for both generators
        
        G_optimizer.step()
        
        # train discriminators ------------------------------------------------
        
        # un-freeze discriminators so that they can be trained
        
        for param in D_A.parameters():
            param.requires_grad = True
            
        for param in D_B.parameters():
            param.requires_grad = True
            
        # zero discriminator parameter gradients
        
        D_optimizer.zero_grad()
        
        # get batches of fake images from the image buffers. Note that half of
        # the images will be from the fake images just recently generated
        # above, and the other half will be from the 50 previously generated
        # images from the buffer. See docs for ImageBuffer() class for details
        
        fake_A = buffer_A.get_batch(fake_A)
        fake_B = buffer_B.get_batch(fake_B)
        
        # make the fake images created by the generators leaf nodes so that
        # the parameters of the generators are not trained
        
        fake_A = fake_A.detach()
        fake_B = fake_B.detach()
        
        # D_A(real_A) for real adversarial loss
        
        D_A_real_out = D_A(real_A)
        
        # the discriminator is trying to classify real images as real
        
        D_A_target = torch.ones_like(D_A_real_out)
        
        D_A_loss_real = adv_loss_func(D_A_real_out,D_A_target)
        
        # D_A(fake_A) for fake adversarial loss
        
        D_A_fake_out = D_A(fake_A)
        
        # the discriminator is trying to classify fake images from the
        # generator as fake
        
        D_A_target = torch.zeros_like(D_A_fake_out)
        
        D_A_loss_fake = adv_loss_func(D_A_fake_out,D_A_target)
        
        # combined loss for discriminator A
        
        D_A_loss = (D_A_loss_real + D_A_loss_fake) / 2
        
        # compute gradients
        
        D_A_loss.backward()
        
        # D_B(real_B) for real adversarial loss
        
        D_B_real_out = D_B(real_B)
        
        # the discriminator is trying to classify real images as real
        
        D_B_target = torch.ones_like(D_B_real_out)
        
        D_B_loss_real = adv_loss_func(D_B_real_out,D_B_target)
        
        # D_B(fake_B) for fake adversarial loss
        
        D_B_fake_out = D_B(fake_B)
        
        # the discriminator is trying to classify fake images from the
        # generator as fake
        
        D_B_target = torch.zeros_like(D_B_fake_out)
        
        D_B_loss_fake = adv_loss_func(D_B_fake_out,D_B_target)
        
        # combined loss for discriminator B
        
        D_B_loss = (D_B_loss_real + D_B_loss_fake) / 2
        
        # compute gradients
        
        D_B_loss.backward()
        
        # update parameters for both discriminators
        
        D_optimizer.step()

    # update learning rates
    
    G_scheduler.step()
    D_scheduler.step()
    
    # show results
                                         
    print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 30)
    epoch_end = time.time()  
    epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end - epoch_start))
    print('Epoch Elapsed Time (HH:MM:SS): ' + epoch_time)
    
    if ((epoch+1) % 5) == 0: # every 5 epochs
        
        # load two random images from the test set
    
        test_imgA,test_imgB = next(iter(test_dataloader))
        
        # put generators in inference mode
        
        G_AB.eval()
        G_BA.eval()
        
        # move images to GPU if available
        
        real_A = test_imgA.to(device)
        real_B = test_imgB.to(device)
        
        # transform images to other domains
        
        fake_A = G_BA(real_B)
        fake_B = G_AB(real_A)
        
        # arrange batch of images along x-axis
        
        real_A = tv.utils.make_grid(real_A,nrow=4,padding=5,normalize=True)
        fake_A = tv.utils.make_grid(fake_A,nrow=4,padding=5,normalize=True)
        real_B = tv.utils.make_grid(real_B,nrow=4,padding=5,normalize=True)
        fake_B = tv.utils.make_grid(fake_B,nrow=4,padding=5,normalize=True)
        
        # arrange batch of images along y-axis
        
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        tv.utils.save_image(tensor = image_grid,
                            fp = 'images_{}epochs.png'.format(epoch+1))
        
        # save weights

        torch.save(G_AB.state_dict(),'G_AB_{}epochs.pt'.format(epoch+1))
        torch.save(G_BA.state_dict(),'G_BA_{}epochs.pt'.format(epoch+1))
        torch.save(D_A.state_dict(),'D_A_{}epochs.pt'.format(epoch+1))
        torch.save(D_B.state_dict(),'D_B_{}epochs.pt'.format(epoch+1))

train_end = time.time()
train_time = time.strftime("%H:%M:%S",time.gmtime(train_end - train_start))
print('\nTotal Training Time (HH:MM:SS): ' + train_time)
