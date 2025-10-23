# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:02:02 2022

@author: Windows
"""

import os
import numpy as np
import pickle
import cv2

raw_data_dir = r"C:\Users\Windows\Desktop\data visualization\Visualization\Sample6\Segmentation"
visualized_data_dir = os.path.join(r"C:\Users\Windows\Desktop\data visualization\Visualization\Sample6", "Segmentation & 2DBBs")

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
                        # in_frame = False
                        # for item in BB2D:
                        #     if 0 <= item[0] <= render_size[0] or 0 <= item[1] <= render_size[1]:
                        #         in_frame = True
                        # if occlusion >= 1:
                        #     in_frame = False
                        
                        # draw bbs
                        color = (0,0,255)
                        thickness = 4
                        
                        # print('all:', BB2D)
                        BB_width = BB2D[1][0] - BB2D[0][0]
                        BB_height = BB2D[1][1] - BB2D[0][1]
                        if in_frame:
                            # print('in frame:', BB2D)
                            start = (int(BB2D[0][0] - 0.05 * BB_width) , int(BB2D[0][1] - 0.05 * BB_height))
                            end = (int(BB2D[1][0] + 0.05 * BB_width) + 1 , int(BB2D[1][1] + 0.05 * BB_height) + 1 )
                            
                            cv2.rectangle(result, start, end, color, thickness)
                        
                        #write image
                        filename = os.path.join(visualized_data_dir, file)
                        # print(filename)
                        cv2.imwrite(filename, result)