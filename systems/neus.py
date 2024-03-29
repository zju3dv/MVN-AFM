import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
import numpy as np
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays, get_afm_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import cv2

def afm_depth_loss(pred_depth, target_depth, depth_mask, far_depth_weight = 0.8):
    
    loss = (pred_depth - target_depth)
    loss[depth_mask] *= 0.0
    loss = torch.mean(torch.pow(loss, 2))
    return loss

def get_cmap():
    cdict = {'red':   [[0.0,  0.0, 0.0],
            [1/3,  88/256, 88/256],
            [2/3,  188/256, 188/256],
            [1.0,  252/256, 252/256]],
            'green': [[0.0,  0.0, 0.0],
            [1/3,  28/256, 28/256],
            [2/3,  128/256, 128/256],
            [1.0, 252/256, 252/256]],
            'blue':  [[0.0,  0.0, 0.0],
            [1/3,  0.0, 0.0],
            [2/3,  0.0, 0.0],
            [1.0,  128/256, 128/256]]}
    return LinearSegmentedColormap('gold', segmentdata=cdict, N=2048)

def get_cmap2():
    cdict = {'red':   [[0.0,  0.0, 0.0],
            [1/4,  3/256, 3/256],
            [3/8,  156/256, 156/256],
            [1/2,  255/256, 255/256],
            [5/8,  255/256, 255/256],
            [3/4,  255/256, 255/256],
            [1.0,  255/256, 255/256]],
            'green': [[0.0,  0.0, 0.0],
            [1/4,  3/256, 3/256],
            [3/8,  51/256, 51/256],
            [1/2,  82/256, 82/256],
            [5/8,  149/256, 149/256],
            [3/4,  233/256, 233/256],
            [1.0, 255/256, 255/256]],
            'blue':  [[0.0,  0.0, 0.0],
            [1/4,  97/256, 97/256],
            [3/8,  108/256, 108/256],
            [1/2,  82/256, 82/256],
            [5/8,  82/256, 82/256],
            [3/4,  108/256, 108/256],
            [1.0,  255/256, 255/256]]}
    return LinearSegmentedColormap('halcyon', segmentdata=cdict, N=2048)

def draw_image(img, cmap, save_dir, label_name, scale_bar):
    plt.imshow(img, cmap=cmap)
    cbar = plt.colorbar(pad=0.02, format='%.2f')
    cbar.set_ticks([np.min(img), np.max(img)])
    cbar.ax.tick_params(labelsize=16) 
    cbar.set_label(label_name, rotation=270, labelpad=0.0, fontsize=18)
    cbar.ax.yaxis.set_label_coords(3.2, 0.5)
    
    plt.axis('off')
    scale_length = 0.1 * 256.0  
    scale_text = scale_bar #'1μm'
    scale_x = 220 
    scale_y = 220
    plt.annotate(
        scale_text,
        xy=(scale_x, scale_y),
        xytext=(scale_x, scale_y - 0.2),
        fontsize=18,
        color='white',
        horizontalalignment='center',
        verticalalignment='center'
    )
    plt.hlines(y=scale_y+14, xmin=scale_x-scale_length/2, xmax=scale_x+scale_length/2, color='white', linewidth=5)
    
    plt.savefig(save_dir, bbox_inches='tight')
    plt.clf()   
@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            
            if self.config.dataset.name == 'AFM':
                rays_o, rays_d = get_afm_rays(directions, c2w)
            else:
                rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1])
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1)
            depth = self.dataset.all_depth_imgs[index, y, x].view(-1)
            depth_mask = self.dataset.all_depth_masks[index, y, x].view(-1)
            gt_depth = self.dataset.all_gt_depth_imgs[index, y, x].view(-1)
        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            if self.config.dataset.name == 'AFM':
                rays_o, rays_d = get_afm_rays(directions, c2w)
            else:
                rays_o, rays_d = get_rays(directions, c2w)
            depth = self.dataset.all_depth_imgs[index].view(-1)
            depth_mask = self.dataset.all_depth_masks[index].view(-1)
            gt_depth = self.dataset.all_gt_depth_imgs[index].view(-1)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1])
            fg_mask = self.dataset.all_fg_masks[index].view(-1)
        
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'depth': depth,
            'gt_depth': gt_depth,
            'fg_mask': fg_mask,
            'depth_mask': depth_mask
        })      
    
    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)   
    
        far_depth_weight = self.config.system.loss.far_depth_weight
        
        
        loss_depth_mse = afm_depth_loss(out['depth_full'][out['rays_valid_full'][...,0]].squeeze(), batch['depth'][out['rays_valid_full'][...,0]].squeeze(), batch['depth_mask'][out['rays_valid_full'][...,0]].squeeze(), far_depth_weight)
        self.log('train/loss_depth', loss_depth_mse)
        loss += loss_depth_mse * self.C(self.config.system.loss.lambda_depth_mse)  

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)  
    

    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        
        depth_mse = 0.0
        
        depth_mse = torch.mean(torch.abs(out['depth'].squeeze().to(batch['gt_depth']) - batch['gt_depth']))
        print(batch['index'][0].item(), psnr.item(), depth_mse.item())

        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        depth_map = out['depth'].view(H, W).cpu().numpy()
        depth_dir = os.path.join(self.save_dir, "it"+str(self.global_step)+"-test", '{:04d}.npz'.format(batch['index'][0].item()))
        
        np.savez(depth_dir, depth_map = depth_map, gt_depth=batch['gt_depth'].view(H, W).cpu().numpy())
        print('depth_map:', depth_map.shape)
        
        error_save_folder = os.path.join(self.save_dir, "error_map")
        os.makedirs(error_save_folder, exist_ok=True)
        color_list = ['jet']
        for color in color_list:
            os.makedirs(os.path.join(error_save_folder, color), exist_ok=True)
            error_map = np.abs(depth_map - batch['gt_depth'].view(H, W).cpu().numpy())
            error_map = error_map*(float(self.config.dataset.real_scan_size)/10.0)
            save_file = os.path.join(error_save_folder, color, '{:04d}.png'.format(batch['index'][0].item()))
            draw_image(error_map, color, save_file, 'Error (μm)', self.config.dataset.scale_bar_str)
        
        depth_save_folder = os.path.join(self.save_dir, "depth_map")
        os.makedirs(depth_save_folder, exist_ok=True)
        
        halcyon_cmap = get_cmap2()
        afm_map = 15.0 - depth_map
        afm_map = afm_map - np.min(afm_map)

        afm_map = afm_map*(float(self.config.dataset.real_scan_size)/10.0)
        
        os.makedirs(os.path.join(depth_save_folder, 'halcyon'), exist_ok=True)
        save_file = os.path.join(depth_save_folder, 'halcyon', '{:04d}.png'.format(batch['index'][0].item()))
        draw_image(afm_map, halcyon_cmap, save_file, 'Height (μm)', self.config.dataset.scale_bar_str)
        save_file = os.path.join(depth_save_folder, 'halcyon', 'raw{:04d}.png'.format(batch['index'][0].item()))
        plt.imsave(save_file, afm_map, cmap=halcyon_cmap)
        
        return {
            'psnr': psnr,
            'depth_mse': depth_mse,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            depth_out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                    depth_out_set[step_out['index'].item()] = {'depth_mse': step_out['depth_mse']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
                        depth_out_set[index[0].item()] = {'depth_mse': step_out['depth_mse'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            depth_mse = torch.mean(torch.stack([o['depth_mse'] for o in depth_out_set.values()]))
            
            mask_save_folder = os.path.join(self.save_dir, "depth_mask")
            os.makedirs(mask_save_folder, exist_ok=True)
            depth_mask = self.dataset.all_depth_masks.cpu().numpy()
            num_images, _, _ = depth_mask.shape
            for i in range(num_images):
                mask = depth_mask[i]
                plt.imshow(mask, cmap='gray')
                plt.axis('off')
                save_file = os.path.join(mask_save_folder, '{:04d}.png'.format(i))
                plt.savefig(save_file)
                plt.clf()
            
            self.export()
            print('Results are saved in:',self.save_dir)
            print('Reconstruction is finished.')
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"model-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )        
