# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:02:02 2022

@author: Windows
"""

import os
import numpy as np
import pickle
import cv2

raw_data_dir = r"C:\Users\Windows\Desktop\data visualization\Dataset\RandomCamera_1_Q4_dronesiviewscom-sep-25-2021-construction-site_V2_Armature"
visualized_data_dir = os.path.join(r"C:\Users\Windows\Desktop\data visualization\Visualization\Sample6", "Keypoints")

for root, dirs, files in os.walk(raw_data_dir):
    # Skip 0_Arch
    if '0_Arch' in root:
        continue
    
    os.chdir(root)
    
    # Filters out the depth maps and semantic segmentation
    if 'Depth Map' in root: continue
    if 'Semantic Segmentation' in root: continue
    
    for file in files:
        # Searching for pickle file and loading it
        if 'Joint_Tracker.pickle' in file:
            with open(file, 'rb') as handle:
                Bone_Capture = pickle.load(handle)
            print('pickle file is loaded')
            # Creating a visualization directory
            os.makedirs(visualized_data_dir, exist_ok=True)
    
    # Iterating in files and handling images
        if ".jpg" in file:
            # Separate base from extension
            base, extension = os.path.splitext(file)
            
            # read image
            print(f'{file} is done')
            img = cv2.imread(file)
            result = img.copy()
            #extract frame from image name
            frame_in_img_name = str(int(base[-4:].strip()))
            
            for key in Bone_Capture[frame_in_img_name]:
                for name in Bone_Capture['workers_name_list']:
                    if name in key:
                        bone_connections = Bone_Capture[frame_in_img_name][key]['bone_connection']
                        bone_locations = Bone_Capture[frame_in_img_name][key]['bone_location_2d']
                        occlusion = Bone_Capture[frame_in_img_name][key]['occlusion']
                        render_size = Bone_Capture[frame_in_img_name]['render_size']
                        bone_name = Bone_Capture[frame_in_img_name][key]['bone_name']
                        bone_location_2d = Bone_Capture[frame_in_img_name][key]['bone_location_2d']
                        
                        for bone_name, bone_pixl_cord in bone_location_2d.items():
                            # check if bone is in frame
                            in_frame = False
                            if 0 <= bone_pixl_cord[0] <= render_size[0] or 0 <= bone_pixl_cord[1] <= render_size[1]:
                                in_frame = True
                            if bone_pixl_cord[2] < 0:
                                in_frame = False
                            if occlusion > 0.95:
                                in_frame = False
                        
                            # draw keypoints
                            color = (0,255,255)
                            circle_radius = 3
                            thickness = -1
                            
                            if in_frame:
                                center = (round(bone_pixl_cord[0]), round(bone_pixl_cord[1]))
                                cv2.circle(result, center, circle_radius, color, thickness)
                        
                        #write image
                        filename = os.path.join(visualized_data_dir, file)
                        cv2.imwrite(filename, result)