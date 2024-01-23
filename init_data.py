import numpy as np
import argparse
import os

def parse_commandline():
    s = argparse.ArgumentParser(description="Perform the end-to-end differentiable blind tip reconstruction from given AFM data.")
    s.add_argument("--input_folder", type=str, default="./afm_data", help="")
    s.add_argument("--data_num", type=int, default=9, help="")
    return s.parse_args()

if __name__=='__main__':
    args = parse_commandline()
    img_len = args.data_num 
    # clean history data
    for i in range(img_len):
        meta_data = np.load(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), allow_pickle=True)
        gt_depth_map = None
        if 'gt_depth_map' in meta_data:
            gt_depth_map = meta_data['gt_depth_map']
        
        if 'errosion_depth' in meta_data:
            if 'last_depth' in meta_data:
                print('save errosion_depth and last_depth')
                np.savez(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), gt_depth_map = gt_depth_map, depth_map = meta_data['depth_map'], errosion_depth = meta_data['errosion_depth'], last_depth = meta_data['last_depth'], extrinsic_mat = meta_data['extrinsic_mat'], mark = meta_data['mark'])
            else:
                print('save errosion_depth')
                np.savez(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), gt_depth_map = gt_depth_map, depth_map = meta_data['depth_map'], errosion_depth = meta_data['errosion_depth'], extrinsic_mat = meta_data['extrinsic_mat'], mark = meta_data['mark'])
        else:
            if 'last_depth' in meta_data:
                print('save last_depth')
                np.savez(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), gt_depth_map = gt_depth_map, depth_map = meta_data['depth_map'], last_depth = meta_data['last_depth'], extrinsic_mat = meta_data['extrinsic_mat'], mark = meta_data['mark'])
            else:
                print('save only depth_map')
                np.savez(os.path.join(args.input_folder, '{:04d}.npz'.format(i)), gt_depth_map = gt_depth_map, depth_map = meta_data['depth_map'], extrinsic_mat = meta_data['extrinsic_mat'], mark = meta_data['mark'])
        print('save:',os.path.join(args.input_folder, '{:04d}.npz'.format(i)))
    print('init data done!')
        
