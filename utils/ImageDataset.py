import torch

# Dataset class used to iterate through data

class ImageDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 img_dir,
                 transforms = None,
                 mode = 'train',
                 aligned = 'True'):
        
        self.img_dir = img_dir
        
        self.transforms = transforms
        
        self.mode = mode
        
        self.pathA = glob(img_dir+'/'+mode+'A/*')
        
        self.pathB = glob(img_dir+'/'+mode+'B/*')
        
        self.aligned = aligned
        
    """
    
    The DataLoader class uses the __len__ method to create the index
    iterator. More specifically, idx in the __getitem__ method ranges from
    0 to __len__(self). Remember this if __len__(self) exceeds the
    length of any lists defined in this class.
    
    See here for details:
        
    https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#SequentialSampler
    
    """    
    
    def __len__(self):
        return min(len(self.pathA),len(self.pathB))

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # the '% len(self.pathA)' is included because the __len__(self)
        # method can return a length that is greater than the length of
        # the self.pathA list. See the note above for details
        
        img_A = Image.open(self.pathA[idx % len(self.pathA)]).convert('RGB')
        
        if self.aligned:
            
            img_B = Image.open(self.pathB[idx % len(self.pathB)]).convert('RGB')
        
        else:
            
            img_B = Image.open(random.choice(self.pathB)).convert('RGB')
        
        img_A = self.transforms(img_A)
        
        img_B = self.transforms(img_B)
                
        return img_A,img_B