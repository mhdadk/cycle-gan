import os

img_dir = 'style_transfer_data/vangogh2photo'

for folder in os.listdir(img_dir):
    for idx,filename in enumerate(os.listdir(os.path.join(img_dir,folder))):
        
        src = os.path.join(img_dir,folder,filename)
        
        # get file extension
        
        ext = os.path.splitext(filename)[1]
        
        dst = os.path.join(img_dir,folder,folder + str(idx)) + ext
        
        os.rename(src,dst)