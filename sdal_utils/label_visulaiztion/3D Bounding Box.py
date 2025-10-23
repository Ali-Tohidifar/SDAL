# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:02:02 2022

@author: Windows
"""

import os
import numpy as np
import pickle
import cv2

raw_data_dir = r"C:\Users\Windows\Desktop\data visualization\Visualization\Sample6\Keypoints"
visualized_data_dir = os.path.join(r"C:\Users\Windows\Desktop\data visualization\Visualization\Sample6", "3DBBs & Keypoints")

for root, dirs, files in os.walk(raw_data_dir):
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
            
            # print(Bone_Capture['workers_name_list'])
            for key in Bone_Capture[frame_in_img_name]:
                for name in Bone_Capture['workers_name_list']:
                    if name in key:
                        bone_connections = Bone_Capture[frame_in_img_name][key]['bone_connection']
                        bone_locations = Bone_Capture[frame_in_img_name][key]['bone_location_2d']
                        render_size = Bone_Capture[frame_in_img_name]['render_size']
                        occlusion = Bone_Capture[frame_in_img_name][key]['occlusion']
                        bone_name = Bone_Capture[frame_in_img_name][key]['bone_name']
                        BB3D = Bone_Capture[frame_in_img_name][key]['BB3D']
                        
                        
                        # check if 3DBB is in frame
                        in_frame = False
                        for item in BB3D:
                            if 0 <= item[0][0] <= render_size[0] or 0 <= item[0][1] <= render_size[1] or 0 <= item[1][0] <= render_size[0] or 0 <= item[1][1] <= render_size[1]:
                                in_frame = True
                            if item[0][2] < 0 or item[1][2] < 0:
                                in_frame = False
                        if occlusion > 0.95:
                            in_frame = False
                        
                        # draw bbs
                        color = (0,0,255)
                        thickness = 2
                        
                        if in_frame:
                            for item in BB3D:
                                start = (int(item[0][0]) , int(item[0][1]))
                                end = (int(item[1][0]) , int(item[1][1]))
                                
                                cv2.line(result, start, end, color, thickness)
                        
                        #write image
                        filename = os.path.join(visualized_data_dir, file)
                        cv2.imwrite(filename, result)