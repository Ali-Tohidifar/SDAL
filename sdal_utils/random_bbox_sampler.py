import os
import random
import shutil
import argparse
import logging
from tqdm import tqdm

"""
Random Bounding Box Sampler

This script randomly selects a specified number of bounding boxes from a YOLO dataset
and copies (or moves) the corresponding image and label files to a new directory. 

The script assumes that the dataset is organized with 'images' and 'labels' subfolders 
within a main training folder. The bounding box information is stored in text files 
within the 'labels' folder, and each text file corresponds to an image file in the 
'images' folder.

Usage:
    The script can be run from the command line with the following arguments:
    
    --input_folder: Path to the main folder containing the 'images' and 'labels' subfolders.
    --output_folder: Path to the destination folder where the selected files will be copied or moved.
    --num_bboxes: The total number of bounding boxes to randomly select from the dataset.
    --log_file: The path to the log file where the script's execution details will be stored (default is 'bbox_selection.log').
    --move: Optional flag to move the selected files instead of copying them. If not specified, files will be copied.

Example:
    python random_bbox_sampler.py --input_folder /path/to/train_folder \
                                   --output_folder /path/to/output_folder \
                                   --num_bboxes 1000 \
                                   --log_file selection.log \
                                   --move

Functions:
    get_bbox_count(file_path): 
        Reads the number of bounding boxes in a label file.
    
    main(train_folder, output_folder, num_bboxes, log_file, move=False): 
        The main function that handles the selection and copying/moving of files.
        It sets up logging, processes the dataset, and handles the file operations.

Arguments:
    train_folder (str): Path to the main folder containing 'images' and 'labels' subfolders.
    output_folder (str): Path to the folder where selected files will be copied or moved.
    num_bboxes (int): The total number of bounding boxes to randomly select.
    log_file (str): The path to the log file where execution details are stored.
    move (bool): If True, move the selected files instead of copying them. Default is False.

Logging:
    The script logs each step of the process, including file operations and any potential warnings 
    (e.g., if an image file corresponding to a label file is not found).

Note:
    - The script creates the necessary subdirectories in the output folder if they don't already exist.
    - The 'labels' and 'images' subfolders must exist in the input folder.
"""


def get_bbox_count(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)

def main(train_folder, output_folder, num_bboxes, log_file, move=False):
    # Set up logging
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    labels_folder = os.path.join(train_folder, 'labels')
    images_folder = os.path.join(train_folder, 'images')

    if not os.path.exists(labels_folder) or not os.path.exists(images_folder):
        logging.error("Train folder should contain 'images' and 'labels' subfolders.")
        return

    output_labels_folder = os.path.join(output_folder, 'labels')
    output_images_folder = os.path.join(output_folder, 'images')

    os.makedirs(output_labels_folder, exist_ok=True)
    os.makedirs(output_images_folder, exist_ok=True)

    # Get all label files
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

    # List to hold tuples of (file_path, bbox_count)
    bbox_list = []

    for label_file in tqdm(label_files):
        file_path = os.path.join(labels_folder, label_file)
        bbox_count = get_bbox_count(file_path)
        for _ in range(bbox_count):
            bbox_list.append(file_path)
    logging.info(f"Total bounding boxes found: {len(bbox_list)}")
    
    # Randomly shuffle and select the required number of bounding boxes
    random.shuffle(bbox_list)
    selected_files = bbox_list[:num_bboxes]

    selected_files_set = set(selected_files)

    for file_path in selected_files_set:
        # Copy label file
        if not move:
            shutil.copy(file_path, output_labels_folder)
            logging.info(f"Copied label file: {file_path}")
        else:
            shutil.move(file_path, output_labels_folder)
            logging.info(f"Moved label file: {file_path}")
        
        # Copy corresponding image file
        image_file_name = os.path.basename(file_path).replace('.txt', '.jpg')
        image_file_path = os.path.join(images_folder, image_file_name)

        if os.path.exists(image_file_path):
            if not move:
                shutil.copy(image_file_path, output_images_folder)
                logging.info(f"Copied image file: {image_file_path}")
            else:
                shutil.move(image_file_path, output_images_folder)
                logging.info(f"Moved image file: {image_file_path}")
        else:
            logging.warning(f"Image file {image_file_path} not found!")

    logging.info(f"Copied {len(selected_files_set)} label files with a total of {num_bboxes} bounding boxes to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly select a number of bounding boxes from a YOLO dataset.")
    parser.add_argument('--input_folder', type=str, help='Path to the folder containing the train subfolders (images and labels)')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where selected files will be copied')
    parser.add_argument('--num_bboxes', type=int, help='Number of bounding boxes to keep')
    parser.add_argument('--log_file', type=str, default='bbox_selection.log', help='Log file to store the process details')
    parser.add_argument('--move', action='store_true', help='Move the selected files instead of copying', default=False)

    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.num_bboxes, args.log_file, args.move)
