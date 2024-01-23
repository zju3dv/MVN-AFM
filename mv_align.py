import open3d as o3d
import numpy as np
import argparse
import os
import torch
from scipy.spatial.transform import Rotation as R
#from evo.core import trajectory
#from evo.core import lie_algebra as lie
import cv2

def init_align(source_points, target_points):
    # init align AFM data according to their coarse point correspondences
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_points)
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)

    num_points = source_points.shape[0]
    correspondences = o3d.utility.Vector2iVector()
    for i in range(num_points):
        correspondences.append([i, i])

    # compute transformation
    reg_p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    transformation = reg_p2p.compute_transformation(source_cloud, target_cloud, correspondences)

    print("init poses:", transformation)
    return transformation

def transform_pointclouds(pcds, frame0_pcd_o3d, Ts = None):
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        if Ts is not None:
            pcds[point_id].transform(Ts[point_id])
            
        if point_id == 0:
            frame0_pcd_o3d.transform(Ts[point_id])
            pcd_combined += frame0_pcd_o3d
        else:
            pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.06)
    return pcd_combined_down

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    # align two set of AFM data by point-to-plane ICP
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 30))
        
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 30))
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    # build pose graph and optimize all poses
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            
            if source_id == 0: # center frame
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else: 
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

def global_align(pcds):
    # align all AFM data
    voxel_size = 0.2
    max_correspondence_distance_coarse = voxel_size * 0.8 
    max_correspondence_distance_fine = voxel_size * 0.5 
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)
    
    # optimize pose graph
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
    Ts = []
    for point_id in range(len(pcds)):
        Ts.append(pose_graph.nodes[point_id].pose)
    return Ts, pcds

def depth2pcd(depth, img_size, scan_height, marker_points_2d, mask = None):
    marker_points_3d = []
    h, w = depth.shape
    pcd = o3d.geometry.PointCloud()
    for i in range(h):
        for j in range(w):
            if mask is not None:
                if mask[i][j] !=0: 
                    continue
            x = (float(i)/float(h)-0.5)*img_size
            y = (float(j)/float(w)-0.5)*img_size
            z = scan_height - depth[i][j]
            pcd.points.append([y, -1*x, z])
    for mark_point_2d in marker_points_2d:
        x = (float(mark_point_2d[0])/float(h)-0.5)*img_size
        y = (float(mark_point_2d[1])/float(w)-0.5)*img_size
        z = scan_height - depth[mark_point_2d[0]][mark_point_2d[1]]
        marker_points_3d.append([y, -1*x, z])
    return pcd, marker_points_3d

def get_afm_ray_directions(H, W, K):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i[...,None].repeat(1,1,3)
    j = j[...,None].repeat(1,1,3)
    vx = torch.tensor([float(K),0,0]).float()
    vy = torch.tensor([0,float(K),0]).float()
    directions = (i/float(H)-0.5) * vy + (j/float(W)-0.5) * vx
    return directions

def get_afm_rays(directions, c2w, keepdim=False):
    # convert AFM data to orthogonal camera ray
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

def save_trajectory_tum_format(filename, traj):
    with open(filename, 'w') as f:
        for ts, pose in zip(traj.timestamps, traj.poses_se3):
            t = pose[:3,3]
            q = R.from_matrix(pose[:3,:3]).as_quat()
            f.write(f"{ts} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")

def dilate_mask(mask, iterations=1):
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=iterations)

    dilated_mask = (dilated_mask_uint8 > 0).astype(bool)

    return dilated_mask

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
    mask = dilate_mask(mask, iterations=2)
    return mask

def parse_commandline():
    s = argparse.ArgumentParser(description="Multiview AFM data alignment")
    s.add_argument("--input_folder", type=str, default="./input", help="multiview data folder")
    s.add_argument("--data_num", type=int, default=9, help="")
    return s.parse_args()

if __name__=='__main__':
    args = parse_commandline()
    pcds = [] 
    marker_points_3d_list = [] 
    meta_data_list = []
    img_len = args.data_num 
    for i in range(img_len):
        
        meta_data = np.load(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), allow_pickle=True)
        meta_data_list.append(meta_data)
        marker_points_2d = meta_data['mark']
        print(marker_points_2d)

        depth_img = meta_data['depth_map']
        mask = None
        if 'mask' in meta_data:
            print('pre mask')
            mask = meta_data['mask']
        else:
            print('no mask')
                
        if i == 0:
            init_mask = init_pcd_mask(depth_img, 10.0, 10.0)
            frame0_pcd_o3d, _ =  pcd_o3d, marker_points_3d = depth2pcd(depth_img, 10.0, 10.0, marker_points_2d, mask=mask)
            mask = init_mask  

        
        pcd_o3d, marker_points_3d = depth2pcd(depth_img, 10.0, 10.0, marker_points_2d, mask=mask)
        
        pcd_o3d.estimate_normals()
        camera_location = np.array([0., 0., 1000.])
        pcd_o3d.orient_normals_to_align_with_direction(camera_location)
        pcds.append(pcd_o3d)

        marker_points_3d_list.append(marker_points_3d)
    
    
    print('marker:',marker_points_3d_list)
    init_Ts = []
    for i in range(img_len):
        source_points = np.asarray(marker_points_3d_list[0])
        target_points = np.asarray(marker_points_3d_list[i])
        
        init_T = init_align(target_points, source_points)
        init_Ts.append(init_T)
    init_pcd_combined = transform_pointclouds(pcds, frame0_pcd_o3d, init_Ts)

    Ts, pcds = global_align(pcds)
    pcd_combined = transform_pointclouds(pcds, frame0_pcd_o3d, Ts)

    base_pose = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,10],[0,0,0,1]])
    
    for i in range(img_len):
        meta_data = meta_data_list[i]
        
        pose = np.matmul(Ts[i], init_Ts[i])
        
        pose = np.matmul(pose, base_pose)
        pose = np.linalg.inv(pose)
        
        gt_depth_map = None
        if 'gt_depth_map' in meta_data:
            gt_depth_map = meta_data['gt_depth_map']


        if 'errosion_depth' in meta_data:
            if 'last_depth' in meta_data:
                print('save errosion_depth and last_depth')
                np.savez(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), gt_depth_map = gt_depth_map, depth_map = meta_data['depth_map'], errosion_depth = meta_data['errosion_depth'], last_depth = meta_data['last_depth'], extrinsic_mat = pose[:3,:], mark = meta_data['mark'])
            else:
                print('save errosion_depth')
                np.savez(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), gt_depth_map = gt_depth_map, depth_map = meta_data['depth_map'], errosion_depth = meta_data['errosion_depth'], extrinsic_mat = pose[:3,:], mark = meta_data['mark'])
        else:
            if 'last_depth' in meta_data:
                print('save last_depth')
                np.savez(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), gt_depth_map = gt_depth_map, depth_map = meta_data['depth_map'], last_depth = meta_data['last_depth'], extrinsic_mat = pose[:3,:], mark = meta_data['mark'])
            else:
                print('save only depth_map')
                np.savez(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), gt_depth_map = gt_depth_map, depth_map = meta_data['depth_map'], extrinsic_mat = pose[:3,:], mark = meta_data['mark'])
        print('save:',os.path.join(args.input_folder, '{:04d}.npz'.format(i)))
        