import torch
from .coco import CoCo
from .transforms import transform
from torchvision.datasets import ImageFolder


def build_loader(cfg):
	if cfg.dataset.name == "coco":
		# Create the COCO dataset
		train_ds = CoCo(cfg, dataType='train2017', annType='captions', is_train=True)
		val_ds = CoCo(cfg, dataType='val2017', annType='captions', is_train=False)


	train_dl = torch.utils.data.DataLoader(train_ds,
											batch_size=cfg.dataset.params.batch_size, 
											shuffle=cfg.dataset.params.shuffle, 
											num_workers=cfg.dataset.params.num_workers)  
	val_dl = torch.utils.data.DataLoader(val_ds,
											batch_size=cfg.dataset.params.batch_size, 
											shuffle=cfg.dataset.params.shuffle, 
											num_workers=cfg.dataset.params.num_workers) 
	
	return (train_dl, val_dl)


			
			
 