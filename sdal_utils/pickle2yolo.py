# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 19:58:29 2023

@author: Windows
"""
import os
import pickle

# generate label
def pickle2yolo(image_folder, output_dir, save_txt = True):
    All_BBs = {}
    Bone_Capture = None
    for file in os.listdir(image_folder):
        if 'Joint_Tracker.pickle' in file:
            label_dir = os.path.join(image_folder, file)
            with open(label_dir, 'rb') as handle:
                Bone_Capture = pickle.load(handle)
    
    if Bone_Capture is None:
        # import ipdb; ipdb.set_trace()
        raise ValueError('No Joint_Tracker.pickle file found in the folder')
    
    for file in os.listdir(image_folder):
        if 'Depth Map' in file: continue
        if 'Semantic Segmentation' in file: continue
        if '.jpg' in file:
            frame_in_img_name = str(int(file.strip('.jpg')[-4:]))
        
            BBs = []
            for key in Bone_Capture[frame_in_img_name]:
                if 'camera_location' in key: 
                    continue
                elif 'render_size' in key: 
                    continue
                else:
                    bone_connections = Bone_Capture[frame_in_img_name][key]['bone_connection']
                    bone_locations = Bone_Capture[frame_in_img_name][key]['bone_location_2d']
                    render_size = Bone_Capture[frame_in_img_name]['render_size']
                    occlusion = Bone_Capture[frame_in_img_name][key]['occlusion']
                    bone_name = Bone_Capture[frame_in_img_name][key]['bone_name']
                    BB2D = Bone_Capture[frame_in_img_name][key]['BB2D']
                    
                    # check if 2DBB is in frame
                    in_frame = False
                    for item in BB2D:
                        if 0 <= item[0] <= render_size[0] and 0 <= item[1] <= render_size[1]:
                            in_frame = True
                    if occlusion >= 1:
                        in_frame = False
                    if BB2D[0][0]*BB2D[0][1] < 0:
                        if BB2D[1][0] > render_size[0] or BB2D[1][1] > render_size[1]:
                            in_frame = True
                    
                    if occlusion >= 1:
                        in_frame = False
        
                    if in_frame:
                        # Calculating required data from bounding box
                        minx = max(min(int(BB2D[0][0]) , render_size[0]) , 0)
                        maxx = max(min(int(BB2D[1][0]) , render_size[0]) , 0)
                        miny = max(min(int(BB2D[0][1]) , render_size[1]) , 0)
                        maxy = max(min(int(BB2D[1][1]) , render_size[1]) , 0)
                        
                        center_x = (minx + maxx + 1) / (2 * render_size[0])
                        center_y = (miny + maxy + 1) / (2 * render_size[1])
                        width = (maxx - minx + 1)/render_size[0]
                        height = (maxy - miny + 1)/render_size[1]
                        
                        # Writing data to txt file
                        BBs.append('0 {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(center_x, center_y, width, height))
            
            if len(BBs) == 0:
                print('No labels for this image:\n', image_folder.stem + '_' + file)
            else:
                BBs[-1] = BBs[-1].strip('\n')
            
            All_BBs[file] = BBs
            
            if save_txt:
                txt_dir = os.path.join(output_dir, os.path.basename(image_folder) + '_' + file.replace('jpg', 'txt'))
                with open(txt_dir, 'w') as f:
                    f.write(''.join(list(BBs)))
    return All_BBs