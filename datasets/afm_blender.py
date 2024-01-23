import os
import json
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions, get_afm_ray_directions
from utils.misc import get_rank

import imageio 
import cv2

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,np.sin(th),0],
    [0,1,0,0],
    [-np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

rot_gamma = lambda ga : torch.Tensor([
    [np.cos(ga),-np.sin(ga), 0, 0],
    [np.sin(ga), np.cos(ga), 0, 0],
    [0,0,1,0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, gamma, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_gamma(gamma/180.*np.pi) @ c2w
    return c2w

class AFMDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = False
        self.apply_mask = False

        img_range = 0
        if self.split == 'train':
            img_range = self.config.train_img_range
        elif self.split == 'val':
            img_range = self.config.val_img_range
        elif self.split == 'test':
            img_range = self.config.test_img_range
        
        self.w, self.h = self.config.img_wh
        self.img_wh = (self.w, self.h)

        self.near, self.far = 2.0, 16.0

        self.focal = self.config.focal #AFM scan size

        self.directions = \
            get_afm_ray_directions(self.w, self.h, self.focal).to(self.rank) # (h, w, 3)           

        self.all_c2w, self.all_images, self.all_gt_depth_imgs, self.all_depth_imgs, self.all_fg_masks, self.all_depth_masks = [], [], [], [], [], []
        init_depth_imgs = []
        
        if self.split == 'test' and self.config.full_renderer:
            render_poses = torch.stack([pose_spherical(0.0, -155.0, angle, -10.0) for angle in np.linspace(-180,180,200+1)[:-1]], 0)
            self.all_c2w = render_poses.float().to(self.rank)
            self.all_images = torch.ones((200, self.w, self.h, 3)).float().to(self.rank)
            self.all_depth_imgs = torch.ones((200, self.w, self.h, 3)).float().to(self.rank)
            self.all_gt_depth_imgs = torch.ones((200, self.w, self.h, 3)).float().to(self.rank)
            self.all_fg_masks = torch.ones((200, self.w, self.h, 3)).float().to(self.rank)
            self.all_depth_masks = torch.ones((200, self.w, self.h, 3)).float().to(self.rank)
            return 
        
        for i in range(img_range):
            
            img = np.ones((self.w, self.h, 3), dtype=np.float32)
            
            self.all_images.append(torch.from_numpy(img[...,:3]))

            meta_data = np.load(os.path.join(self.config.root_dir, '{:04d}.npz'.format(i)))
            depth_img = meta_data['depth_map']

            init_depth_imgs.append(torch.from_numpy(depth_img))
            
            depth_img = torch.from_numpy(depth_img)
            self.all_depth_imgs.append(depth_img)

            gt_depth_img = depth_img
            gt_depth_mask = np.zeros_like(depth_img, bool)
            gt_depth_mask = meta_data['mask']
            
            if False:
                gt_depth_img = meta_data['gt_depth_map']
                gt_depth_img = torch.from_numpy(gt_depth_img)
            else:
                gt_depth_img = depth_img
                
            
            self.all_gt_depth_imgs.append(gt_depth_img)
           
            self.all_depth_masks.append(torch.from_numpy(gt_depth_mask))

            print(i,torch.mean((depth_img - gt_depth_img)**2))
            
            pose = meta_data['extrinsic_mat']
            # 3x4 -> 4x4
            pose = np.concatenate([pose, np.array([[0,0,0,1]])], 0)
            pose = np.linalg.inv(pose) 
            pose = torch.from_numpy(pose[:3, :4])
            self.all_c2w.append(pose)

            fg_mask = torch.ones_like(depth_img)
            self.all_fg_masks.append(fg_mask)

        self.all_c2w, self.all_images, self.all_depth_imgs, self.all_gt_depth_imgs, self.all_fg_masks, self.all_depth_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_depth_imgs, dim=0).float().to(self.rank), \
            torch.stack(self.all_gt_depth_imgs, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank), \
            torch.stack(self.all_depth_masks, dim=0).to(self.rank)
        
        init_depth_imgs = torch.stack(init_depth_imgs, dim=0).float().to(self.rank)
        
        print('all_c2w', self.all_c2w.shape)
        print('all_images', self.all_images.shape)
        print('all_fg_masks', self.all_fg_masks.shape)
        print('all_depth_imgs', self.all_depth_imgs.shape)
        print('all_gt_depth_imgs', self.all_gt_depth_imgs.shape)
        print('all_depth_masks', self.all_depth_masks.shape)
        print(self.split,'Depth MSE:', torch.mean((self.all_depth_imgs - self.all_gt_depth_imgs)**2))
        
        if self.split == 'test':
            self.input_depth_mse = torch.mean(torch.abs(self.all_depth_imgs - self.all_gt_depth_imgs)).cpu().numpy()
            
            print('input_depth_mse', self.input_depth_mse)
        
        

class AFMDataset(Dataset, AFMDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class AFMIterableDataset(IterableDataset, AFMDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('AFM')
class AFMDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = AFMIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = AFMDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = AFMDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = AFMDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
