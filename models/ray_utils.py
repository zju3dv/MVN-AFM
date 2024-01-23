import torch
import numpy as np


def cast_rays(ori, dir, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

def get_afm_ray_directions(H, W, K):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i[...,None].repeat(1,1,3)
    j = j[...,None].repeat(1,1,3)
    vx = torch.tensor([float(K),0,0]).float()
    vy = torch.tensor([0,float(K),0]).float()
    directions = (i/float(H)-0.5) * vy + (j/float(W)-0.5) * vx
    return directions


def get_afm_rays(directions, c2w, keepdim=False):
    assert directions.shape[-1] == 3
    look_at = torch.tensor([0,0,1]).float().to(c2w.device)

    if directions.ndim == 2:
        assert c2w.ndim == 3 
        plane_center = c2w[:,:3,-1]
        rays_o = plane_center.expand(directions.shape)
        
        rays_o = rays_o[...,None] + torch.matmul(c2w[:,:3,:3], directions[:,:,None])
        
        rays_o = rays_o.squeeze()
        
        look_at = torch.matmul(c2w[:,:3,:3], look_at)
        rays_d = look_at.expand(rays_o.shape)
    elif directions.ndim == 3: 
        if c2w.ndim == 2: 
            look_at = torch.matmul(c2w[:3,:3], look_at)
            plane_center = c2w[:3,-1]
            rays_o = plane_center.repeat(directions.shape[0],directions.shape[1],1)
            rays_o = rays_o + (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1)
            rays_d = look_at.expand(rays_o.shape)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    return rays_o, rays_d


if __name__=='__main__':
    directions = torch.rand(15, 3)
    c2w = torch.rand(15, 3, 4)
    rays_o, rays_d = get_afm_rays(directions, c2w)
    print(rays_o.shape, rays_d.shape)