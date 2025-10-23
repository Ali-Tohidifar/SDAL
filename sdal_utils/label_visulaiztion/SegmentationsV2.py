import os
import argparse
import numpy as np
import cv2

ALPHA = 0.6
RGBFORLABELS = {
    1: (255,0,0),       # Red
    2: (255,128,0),     # Orange
    3: (255,255,0),     # Yellow
    4: (128,255,0),     # Light green-yellow
    5: (0,255,0),       # Green
    6: (0,255,128),     # Aquamarine
    7: (0,255,255),     # Cyan
    8: (0,128,255),     # Sky blue
    9: (0,0,255),       # Blue
    10: (128,0,255),    # Violet
    11: (255,0,255),    # Magenta
    12: (255,0,128),    # Deep pink
    13: (192,192,192),  # Silver
    14: (128,128,128),  # Gray
    15: (0,128,128),    # Teal
    16: (128,0,0),      # Maroon
    17: (128,128,0),    # Olive
    18: (0,0,128),      # Navy
    19: (139,69,19),    # Saddle brown
    20: (244,164,96),   # Sandy brown
    21: (250,128,114),  # Salmon
    22: (85,107,47),    # Dark olive green
    23: (107,142,35),   # Olive drab
    24: (199,21,133),   # Medium violet red
    25: (70,130,180),   # Steel blue
    26: (153,50,204),   # Dark orchid
    27: (178,34,34),    # Fire brick
    28: (189,183,107),  # Dark khaki
    29: (255,140,0),    # Dark orange
    30: (72,209,204)    # Medium turquoise
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize segmentation masks')
    parser.add_argument('--raw_data_dir', type=str, default=r'sdal_utils\Data_Generator\Dataset_used\Digging_Mixamo_V2_1', help='Directory containing raw images')
    parser.add_argument('--visualized_data_dir', type=str, default=r'C:\Users\Windows\OneDrive - University of Toronto\UofT_PhD\Research Projects\21-09-21_Synthetic_Data\6_Paper1_Trainability-Scalability-Instantiation\0_Support\data visualization\SegmentationVisualization' , help='Directory to save visualized images')
    
    args = parser.parse_args()

    raw_data_dir = args.raw_data_dir
    visualized_data_dir = args.visualized_data_dir

    src_imgs = []
    segment_imgs = []

    for file in os.listdir(raw_data_dir):
        if file in ['Depth Map', 'Semantic Segmentation'] or ".jpg" not in file: 
            continue
        elif ".jpg" in file:
            src_imgs.append(file)
    print('Source images are loaded')
    
    for file in os.listdir(os.path.join(raw_data_dir, 'Semantic Segmentation')):
        if ".jpg" in file: 
            segment_imgs.append(file)

    print('Segmentation images are loaded')
    os.makedirs(visualized_data_dir, exist_ok=True)
    
    print(f'Number of source images: {len(src_imgs)}')
    print(f'Number of segmentation images: {len(segment_imgs)}')
    
    background = 50
    ranges = [i for i in range(254, 74, -10)]

    for src_img_name, seg_img_name in zip(src_imgs, segment_imgs):
        main_img = cv2.imread(os.path.join(raw_data_dir, src_img_name))
        seg_img = cv2.imread(os.path.join(raw_data_dir, 'Semantic Segmentation', seg_img_name), cv2.IMREAD_GRAYSCALE)

        mask_overlay = np.zeros_like(main_img)

        for i, pixel_val in enumerate(ranges):
            if pixel_val == background:
                continue
            label = i + 1
            # mask = seg_img == pixel_val
            mask = (seg_img >= pixel_val-5) & (seg_img <= pixel_val+5)
            colour = RGBFORLABELS.get(label)
            if colour is not None:
                mask_overlay[mask] = colour

        result = cv2.addWeighted(main_img, 1, mask_overlay, ALPHA, 0)
        
        filename = os.path.join(visualized_data_dir, src_img_name)
        cv2.imwrite(filename, result)
        print(f'{src_img_name} is done')




    # ranges = [(i-4, i + 4) for i in range(259, 1, -10)]
    # ranges[-1] = (0.5, 4)

    # for src_img_name, seg_img_name in zip(src_imgs, segment_imgs):
    #     main_img = cv2.imread(os.path.join(raw_data_dir, src_img_name))
    #     seg_img = cv2.imread(os.path.join(raw_data_dir, 'Semantic Segmentation', seg_img_name), cv2.IMREAD_GRAYSCALE)

    #     mask_overlay = np.zeros_like(main_img)

    #     for i, (lower, upper) in enumerate(ranges):
    #         if lower <= 50 and upper >= 50:
    #             continue
    #         label = i + 1
    #         mask = (seg_img >= lower) & (seg_img <= upper)
    #         colour = RGBFORLABELS.get(label)
    #         if colour is not None:
    #             mask_overlay[mask] = colour

    #     alpha = 0.4
    #     result = cv2.addWeighted(main_img, 1, mask_overlay, alpha, 0)
        
    #     filename = os.path.join(visualized_data_dir, src_img_name)
    #     cv2.imwrite(filename, result)
    #     print(f'{src_img_name} is done')

    print('Visualization complete')