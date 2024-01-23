import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.autograd import grad
from torch import nn
import time
import copy
import open3d as o3d
from tqdm import trange
import cv2


def convert_mesh(pcd_np, triangles):
    # create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertices = o3d.utility.Vector3dVector(pcd_np)
    mesh.compute_vertex_normals()
    #o3d.visualization.draw_geometries([mesh])
    return mesh

def compute_triangles(img_h, img_w):
    triangles = []
    for i in range(0,img_h-1):
        for j in range(0,img_w-1):
            idx0 = i*img_w + j
            idx1 = i*img_w + j + 1
            idx2 = (i+1)*img_w + j
            idx3 = (i+1)*img_w + j + 1
            triangles.append([idx0, idx2, idx1])
            triangles.append([idx1, idx2, idx3])
    return triangles

def compute_triangles_mask(img_h, img_w, mask):
    triangles = []
    for i in range(0,img_h-1):
        for j in range(0,img_w-1):
            idx0 = i*img_w + j
            idx1 = i*img_w + j + 1
            idx2 = (i+1)*img_w + j
            idx3 = (i+1)*img_w + j + 1
            if mask[i, j] or mask[i, j+1] or mask[i+1, j] or mask[i+1, j+1]:
                continue
            triangles.append([idx0, idx2, idx1])
            triangles.append([idx1, idx2, idx3])
    return triangles

def project_blender_image(image, image_pose, view_pose, mesh_base, mesh = None):
    
    depth_image = 15.0 - image
    H = image.shape[0]
    W = image.shape[1]  
    img_rays_o , img_rays_d  = get_afm_rays(H, W, 10.0, image_pose)
    view_rays_o, view_rays_d = get_afm_rays(H, W, 10.0, view_pose)
    
    if mesh is None:
        points = img_rays_o + img_rays_d * np.repeat(depth_image[...,None],3,axis=2)
        # reshape points from W*H*3 to WH * 3
        pcd_np = points.reshape(-1,3)
        mesh = mesh_base
        mesh.vertices = o3d.utility.Vector3dVector(pcd_np)
        mesh.compute_vertex_normals()

    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh)
    # ray cast
    ray = np.concatenate((view_rays_o, view_rays_d), axis=2)
    
    rays = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)

    image_projection = ans['t_hit'].numpy()
    return image_projection

def get_afm_ray_directions(H, W, K):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i[...,None].repeat(1,1,3)
    j = j[...,None].repeat(1,1,3)
    vx = torch.tensor([float(K),0,0]).float()
    vy = torch.tensor([0,float(K),0]).float()
    directions = (i/float(H)-0.5) * vy + (j/float(W)-0.5) * vx
    return directions

def get_afm_rays(directions, c2w, keepdim=False):
    c2w = torch.tensor(c2w).float()
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

def parse_commandline():
    s = argparse.ArgumentParser(description="Multiview AFM data cross-validation and mask solving")
    s.add_argument("--input_folder", type=str, default="./afm_data", help="")
    s.add_argument("--threshold", type=float, default=0.3, help="")
    s.add_argument("--data_num", type=int, default=9, help="")
    return s.parse_args()

def plot_histogram(matrix, interval):
    flattened = matrix.flatten()
    flattened=flattened[flattened>0.01]

    hist, bins = np.histogram(flattened, bins=np.arange(min(flattened), max(flattened) + interval, interval))

    plt.hist(flattened, bins=bins, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Value Distribution Histogram')
    plt.show()


def init_pcd_mask(depth, img_size, scan_height):
    h, w = depth.shape
    pcd_o3d = o3d.geometry.PointCloud()
    for i in range(h):
        for j in range(w):
            x = (float(i)/float(h)-0.5)*img_size
            y = (float(j)/float(w)-0.5)*img_size
            z = scan_height - depth[i][j]
            pcd_o3d.points.append([y, -1*x, z])
    pcd_o3d.estimate_normals()
    camera_location = np.array([0., 0., 10000.])
    pcd_o3d.orient_normals_to_align_with_direction(camera_location)
    normals = np.asarray(pcd_o3d.normals)
    normals = normals.reshape(depth_img.shape[0], depth_img.shape[1], 3)
    z_axis = np.array([0., 0., 1.])
    cos_theta = np.sum(normals * z_axis, axis=2)
    theta = np.arccos(cos_theta)
    theta = theta / np.pi * 180
    theta = theta.astype(np.uint8)
    mask = np.zeros_like(theta)
    mask[theta>70] = 255
    return mask

def expand_matrix(matrix, a=1):
    shape = matrix.shape
        
    new_shape = (shape[0] + 2*a, shape[1] + 2*a) + shape[2:]
    expanded_matrix = np.empty(new_shape, dtype=object)
    
    expanded_matrix[a:a+shape[0], a:a+shape[1]] = matrix
    
    expanded_matrix[:a, a:a+shape[1]] = matrix[0, :]
    expanded_matrix[a+shape[0]:, a:a+shape[1]] = matrix[-1, :]
    expanded_matrix[a:a+shape[0], :a] = matrix[:, 0:1]
    expanded_matrix[a:a+shape[0], a+shape[1]:] = matrix[:, -1:]
    
    expanded_matrix[:a, :a] = matrix[0, 0]
    expanded_matrix[:a, a+shape[1]:] = matrix[0, -1]
    expanded_matrix[a+shape[0]:, :a] = matrix[-1, 0]
    expanded_matrix[a+shape[0]:, a+shape[1]:] = matrix[-1, -1]
    
    exp_len = 1.0
    if len(shape)==3:
        vec_x = np.array([1.0, 0.0, 0.0])
        vec_y = np.array([0.0, 1.0, 0.0])
        expanded_matrix[:a, a:a+shape[1]] += vec_y*exp_len
        expanded_matrix[a+shape[0]:, a:a+shape[1]] -= vec_y*exp_len
        expanded_matrix[a:a+shape[0], :a] -= vec_x*exp_len
        expanded_matrix[a:a+shape[0], a+shape[1]:] += vec_x*exp_len
        expanded_matrix[:a, :a] += (vec_y-vec_x)*exp_len
        expanded_matrix[:a, a+shape[1]:] += (vec_y+vec_x)*exp_len
        expanded_matrix[a+shape[0]:, :a] += (-1.0*vec_y-vec_x)*exp_len
        expanded_matrix[a+shape[0]:, a+shape[1]:] += (-1.0*vec_y+vec_x)*exp_len
    return expanded_matrix

if __name__ == "__main__":
    args = parse_commandline()
    
    img_len = args.data_num 
    meshs = []
    poses = []
    H = 256
    W = 256
    triangles = compute_triangles(H, W)
    mesh_base = o3d.geometry.TriangleMesh()
    mesh_base.triangles = o3d.utility.Vector3iVector(triangles)
    directions = get_afm_ray_directions(256, 256, 10.0)
    ray_o_list = []
    ray_d_list = []
    depths = []
    
    meta_data = np.load(os.path.join(args.input_folder, '{:04d}.npz'.format(0)), allow_pickle=True)
    depth_img = meta_data['depth_map']
    init_mask = init_pcd_mask(depth_img, 10.0, 10.0)
    expand_size = 1
    exp_init_mask = expand_matrix(init_mask, expand_size)
        
    for i in range(img_len):
        meta_data = np.load(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), allow_pickle=True)
        pose = np.concatenate([meta_data['extrinsic_mat'], np.array([[0,0,0,1]])], 0)
        poses.append(pose)
        
        depth = meta_data['depth_map']
        depths.append(depth)
        depth = torch.tensor(depth).float()

        rays_o, rays_d = get_afm_rays(directions, np.linalg.inv(pose), keepdim=True)
        ray_o_list.append(rays_o)
        ray_d_list.append(rays_d)
        points = rays_o + rays_d * depth[...,None].repeat(1,1,3)
        points = points.cpu().numpy()
        print('points:', points.shape)
        
        mesh = None
        if i == 0:
            triangles = compute_triangles_mask(H+2*expand_size, W+2*expand_size, exp_init_mask)
            
            points = expand_matrix(points, expand_size)
            print('points:', points.shape)
            mesh = o3d.geometry.TriangleMesh()
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
        else:
            mesh = copy.deepcopy(mesh_base)
        pcd_np = points.reshape(-1,3)
        mesh.vertices = o3d.utility.Vector3dVector(pcd_np)
        mesh.compute_vertex_normals()
        
        meshs.append(mesh)
    
    masks = []
    
    for i in trange(img_len):
        tmp_mask = np.zeros((H,W), bool)
        view_rays_o, view_rays_d = ray_o_list[i], ray_d_list[i]

        ray = np.concatenate((view_rays_o, view_rays_d), axis=2)
        
        rays = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)
        
        if i == 0:
            tmp_mask = init_mask
        
        for j in range(0, img_len): 
            if i == j:
                continue
            scene = o3d.t.geometry.RaycastingScene()
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(meshs[j])
            scene.add_triangles(mesh)
            
            ans = scene.cast_rays(rays)

            image_projection = ans['t_hit'].numpy()
            image_projection[np.isinf(image_projection)]=0
            
            tmp_mask = np.logical_or(tmp_mask, image_projection - depths[i] > args.threshold)
            
        masks.append(tmp_mask)
    
    print('save mask...')
    
    for i in trange(img_len):
        meta_data = np.load(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), allow_pickle=True)
        gt_depth_map = None
        if 'gt_depth_map' in meta_data:
            gt_depth_map = meta_data['gt_depth_map']
        np.savez(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), gt_depth_map = gt_depth_map, depth_map = meta_data['depth_map'], extrinsic_mat = meta_data['extrinsic_mat'], mark = meta_data['mark'], mask = masks[i])
        
