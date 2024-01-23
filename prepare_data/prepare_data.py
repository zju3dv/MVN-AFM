from tkinter import *
from tkinter.colorchooser import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import copy
import imageio 
import argparse
import shutil 

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def numpy_to_photo(img):
    # convert numpy array to gray image
    img = to8b(img)
    im = Image.fromarray(img)
    photo = ImageTk.PhotoImage(image=im)
    return photo

def init_window(w = 500, h = 500):
    # initialize the GUI window
    root = Tk()
    root.title("Marker Chooser")
    root.geometry("500x500")
    root.resizable(False,False)
    root.config(background="white")
    width  = w+2
    height = h+2
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    size_geo = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)
    root.geometry(size_geo)
    return root

def draw_markers(img, xy_list):
    # draw a circle at the location of the marker
    img = copy.deepcopy(img)
    img = to8b(img)
    for idx in range(len(xy_list)):
        i = int(xy_list[idx][0])
        j = int(xy_list[idx][1])
        if idx == 0:
            color = (255, 0, 0)
        elif idx == 1:
            color = (0, 255, 0)
        elif idx == 2:
            color = (0, 0, 255)
        cv2.circle(img, (j,i), 5, color, -1)
    return img

def get_marker_location(event):
    # read and save the location of the marker from the mouse click
    global xy_list, output_folder, curr_idx, img_len, img_list
    x, y = event.x, event.y
    xy_list.append([y,x])
    if len(xy_list)>=3:

        meta_data = np.load(os.path.join(output_folder, '{:04d}.npz'.format(curr_idx)), allow_pickle=True)
        marker_points_2d = np.array(xy_list, dtype=np.int32)
        print('xy:',marker_points_2d)
        np.savez(os.path.join(output_folder, '{:04d}.npz'.format(curr_idx)), depth_map = meta_data['depth_map'], extrinsic_mat = meta_data['extrinsic_mat'], mark = marker_points_2d)

        img = draw_markers(img_list[curr_idx], xy_list)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_folder, 'mark_{:04d}.png'.format(curr_idx)), img)
        xy_list = []
        print('save image', curr_idx)
        print()
        curr_idx += 1
        if curr_idx >= img_len:
            print('all done')
            exit()
        update_img(curr_idx)    
        

def parse_commandline():
    s = argparse.ArgumentParser(description="Perform the end-to-end differentiable blind tip reconstruction from given AFM data.")
    s.add_argument("--input_folder", type=str, default="./input", help="")
    s.add_argument("--output_folder", type=str, default="./output", help="")
    s.add_argument("--scan_range", type=float, help="the AFM scanning range of your images(μm)")
    s.add_argument("--data_num", type=int, default=9, help="the number of AFM multiview scanning")
    return s.parse_args()

def load_txt(file_dir, scan_range = 1.0):
    # read the AFM data matrix from txt file
    all_data = []
    with open(file_dir, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            all_data.append(line.strip().split())
    all_data = np.array(all_data, dtype=np.float64)
    all_data*=(10e6/scan_range)#(10e6/2.0) #normal 1e6 pmma 5e6 MOF #10e6/1.5
    # print(all_data)
    # print(all_data.shape)
    return all_data

def make_rgbd_img(depth_data):
    depth_img = ((np.max(depth_data) - depth_data) / (np.max(depth_data)-np.min(depth_data)) * 255 ).astype(np.uint8)
    rgb_img = ((depth_data - np.min(depth_data)) / (np.max(depth_data)-np.min(depth_data)) * 255 ).astype(np.uint8)
    depth_img = depth_img[...,None].repeat(3, axis=2)
    rgb_img = rgb_img[...,None].repeat(3, axis=2)
    return rgb_img, depth_img

def update_img(idx):
    # Display afm images in sequence
    global img_list
    img = img_list[idx]
    photo = numpy_to_photo(img)
    lab.configure(image=photo)
    lab.image = photo

if __name__=='__main__':
    args = parse_commandline()
    output_folder = args.output_folder

    depth_data = load_txt(os.path.join(args.input_folder, '{:02d}.txt'.format(0)), scan_range=args.scan_range)
    #print(depth_data.shape)
    rgb_img, depth_img = make_rgbd_img(depth_data)
    img = (np.array(depth_img) / 255.).astype(np.float32)

    root = init_window(depth_data.shape[0], depth_data.shape[1])
    photo = numpy_to_photo(img)
    lab = Label(root, image=photo)
    lab.borderwidth = 0
    lab.place(x = 0, y = 0)

    os.makedirs(args.output_folder, exist_ok=True)

    img_len = args.data_num #9
    xy_list = []
    img_list = []
    curr_idx = 0
    for i in range(img_len):
        depth_data = load_txt(os.path.join(args.input_folder, '{:02d}.txt'.format(i)), scan_range=args.scan_range)
        rgb_img, depth_img = make_rgbd_img(depth_data)
        cv2.imwrite(os.path.join(args.output_folder, '{:04d}.png'.format(i)), rgb_img)
        
        depth_data = depth_data - np.max(depth_data)
        
        pose = np.eye(4)
        np.savez(os.path.join(args.output_folder, '{:04d}.npz'.format(i)), depth_map = 10.0 - depth_data, extrinsic_mat = pose[:3,:])

        img = (np.array(rgb_img) / 255.).astype(np.float32)
        img_list.append(img)
    
    print('Please click on the same three points in all images.')
    
    update_img(curr_idx)

    lab.bind('<Button-1>', get_marker_location)
    root.mainloop()

