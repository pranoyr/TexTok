import os
import torch
import zipfile
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from .transforms import transform



class CoCo:
    def __init__(self, root, dataType='train2017', annType='captions', is_train=True):
        
        # if is_train:
        #     root = train_path
        # else:
        #     root = val_path

        
        self.img_dir = '{}/{}'.format(root, dataType)
        annFile = '{}/annotations/{}_{}.json'.format(root, annType, dataType)
        self.coco = COCO(annFile)
        self.imgids = self.coco.getImgIds()
        self.transform = transform(is_train=is_train)
        
        # if cfg.experiment.max_train_examples < len(self.imgids):
        #     self.imgids = self.imgids[:cfg.experiment.max_train_examples]
            
    def __getitem__(self, idx):
        imgid = self.imgids[idx]
        img_name = self.coco.loadImgs(imgid)[0]['file_name']
        annid = self.coco.getAnnIds(imgIds=imgid)
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        ann = np.random.choice(self.coco.loadAnns(annid))['caption']
        
        if self.transform is not None:
            img = self.transform(img)
    
        return img, ann     
        
    def __len__(self):
        return len(self.imgids)



def get_coco_loaders(root, batch_size, num_workers):

    # Create the COCO dataset
    train_ds = CoCo(root, dataType='train2017', annType='captions', is_train=True)
    val_ds = CoCo(root, dataType='val2017', annType='captions', is_train=False)


    train_dl = torch.utils.data.DataLoader(train_ds,
                                            batch_size=batch_size, 
                                            # shuffle=True, 
                                            num_workers=num_workers)  
    val_dl = torch.utils.data.DataLoader(val_ds,
                                            batch_size=batch_size, 
                                            # shuffle=shuffle, 
                                            num_workers=num_workers) 
	
    return (train_dl, val_dl)


			