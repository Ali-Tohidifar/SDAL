import os
import shutil
import random
from tqdm import tqdm
import os
import h5py
import torch
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
# import pickle5 as pickle
import pickle
import yaml
import shutil
import random
import matplotlib.pyplot as plt
import torch
import csv
import json
import re
from pathlib import Path
from blender_depended_codes.normalize_avatar_location import normalize_all_location

from yolov7.test_sdal import test



class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    

def move_imgs(dataset, dst_folder, seq_numbers=500):
    # Create a list of folders with 'RandomCamera' in their names
    img_folders = [folder for folder in os.listdir(dataset) if 'RandomCamera' in folder]
    
    # Randomly select 'seq_numbers' folders from the list
    rnd_folders = random.sample(img_folders, seq_numbers)
    
    # Initialize the progress bar
    pbar = tqdm(total=seq_numbers, desc="Copying files", unit="file")
    
    # Loop over the randomly selected folders
    for file in rnd_folders:
        # Construct the destination path
        dst = os.path.join(dst_folder, file)
        
        # Check if the destination folder does not already exist
        if not os.path.exists(dst):
            # Construct the source path
            src = os.path.join(dataset, file)
            
            # Update the progress bar description with the current file being copied
            pbar.set_description(f"Copying {os.path.basename(src)}")
            pbar.refresh()
            
            # Copy the entire folder tree from source to destination
            shutil.copytree(src, dst)
            
        # Increment the progress bar
        pbar.update(1)
    
    # Close the progress bar
    pbar.close()


def load_image_paths_from_folder(folder_path):
    """
    Retrieve image paths from a folder, excluding specific subfolders.
    """
    print("Starting to retrieve image paths...")

    exclude_folders = {'Depth Map', 'Semantic Segmentation'}
    image_paths = []

    for subdir, dirs, files in tqdm(os.walk(folder_path), desc="Directories", unit="dir"):
        dirs[:] = [d for d in dirs if d not in exclude_folders]
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(subdir, file)
                image_paths.append(file_path)

    print("Image path retrieval completed.")
    return image_paths


# Function to copy and rename files
def copy_and_rename_files(file_paths, destination):
    for file_path in file_paths:
        # Extract the parent directory name
        parent_dir_name = os.path.basename(os.path.dirname(file_path))

        # Construct the new file name
        new_file_name = parent_dir_name + '.jpg'

        # Construct the full destination path
        dest_path = os.path.join(destination, new_file_name)

        # Copy and rename the file
        shutil.copy(file_path, dest_path)
        print(f"Copied and renamed: {file_path} to {dest_path}")


def get_avatar_action_from_name(worker_name):
    """
    Extracts the avatar action and name from the given worker name.

    Args:
        worker_name (str): The name of the worker.

    Returns:
        tuple: A tuple containing the avatar action and name extracted from the worker name.
    """
    splited_name_list = worker_name.split('_')
    special_case = False
    
    for i, word in enumerate(splited_name_list):
        if 'v' in word.lower() and i < len(splited_name_list)-1:
            index = worker_name[worker_name.lower().find('v'):].find('_') + worker_name.lower().find('v')
            action = ''.join(worker_name[:index])
            name = ''.join(worker_name[index+1:]).replace('.001', '').replace('hands_V1_', '')
            special_case = True
    
    if not special_case:
        action = splited_name_list[0]
        name = '_'.join(splited_name_list[1:]).replace('.001', '').replace('hands_V1_', '')
    
    return action, name


def find_matching_avatar(avatars_in_scene, available_avatars, 
                         action2action_json_dir='./sdal_utils/Data_Generator/action2action.json'):
    """
    Finds the best matching avatar based on the specified conditions.

    Args:
        avatars_in_scene (list): A list of avatars in the scene.
        available_avatars (list): A list of available avatars.
        action2action_json_dir (str, optional): The directory of the action2action JSON file. 
            Defaults to './sdal_utils/Data_Generator/action2action.json'.

    Returns:
        str: The best matching avatar.

    Raises:
        ValueError: If no avatar is found with the specified conditions.
    """
    
    with open(action2action_json_dir, 'r') as f:
        action2action = json.load(f)
    
    for worker_name in avatars_in_scene:
        action, name = get_avatar_action_from_name(worker_name)
        alternative_actions = action2action[action]

        # check if matching avatar is available
        avatar_match = {}
        for avatar in available_avatars:
            alternative_found = (False, False) #(name, alternative_actions)
            if name in avatar and alternative_actions in avatar:
                alternative_found = (True, True)
            elif name in avatar and alternative_actions not in avatar:
                alternative_found = (True, False)
            elif name not in avatar and alternative_actions in avatar:
                alternative_found = (False, True)
            elif name not in avatar and alternative_actions not in avatar:
                alternative_found = (False, False)
            avatar_match[avatar] = alternative_found
        
        # find the best matching avatar
        alternative_avatar = None
        for avatar, alternative_found in avatar_match.items(): 
            if alternative_found[0] and alternative_found[1]: # matching both name and alternative actions
                alternative_avatar = avatar
                break
        if alternative_avatar is None:
            for avatar, alternative_found in avatar_match.items():
                if not alternative_found[0] and alternative_found[1]: # matching alternative actions
                    alternative_avatar = avatar
                    break
        if alternative_avatar is None:
            raise ValueError("No avatar found with the specified conditions.")
    
    return alternative_avatar


import threading
import queue

def save_features_worker(features_queue, hdf5_path, feature_shape, image_paths):
    """
    Worker function to save features to the HDF5 file.
    Args:
        features_queue (queue.Queue): Queue containing features to be saved.
        hdf5_path (str): Path to the HDF5 file where features will be saved.
        feature_shape (tuple): Shape of the features.
        image_paths (list): List of image paths.
    """
    with h5py.File(hdf5_path, 'w') as h5f:
        # Create datasets to store image paths and features
        h5f.create_dataset('image_paths', data=image_paths, dtype=h5py.string_dtype(), compression="gzip")
        features_dset = h5f.create_dataset('features', shape=(len(image_paths), *feature_shape), dtype='float32', compression="gzip")
        
        start_idx = 0
        while True:
            features_batch = features_queue.get()
            if features_batch is None:
                break
            end_idx = start_idx + features_batch.shape[0]
            features_dset[start_idx:end_idx] = features_batch
            start_idx = end_idx

def extract_and_save_features_batchwise(image_paths, model, transform, device, batch_size, hdf5_path):
    """
    Extract features in batches and save them incrementally to an HDF5 file.
    Args:
        image_paths (list): List of image paths.
        model (torch model): Model to use for feature extraction.
        transform (torchvision.transforms): Transformations to be applied to the images.
        device (torch.device): Device to run the model on.
        batch_size (int): Number of images to process in a batch.
        hdf5_path (str): Path to the HDF5 file where features will be saved.
    """
    dataset = CustomImageDataset(image_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize feature_shape based on the model's output
    dummy_input = torch.zeros((1, 3, 480, 480)).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    feature_shape = dummy_output.shape[1:]

    features_queue = queue.Queue(maxsize=10)  # Limiting the queue size to avoid memory overload
    saver_thread = threading.Thread(target=save_features_worker, args=(features_queue, hdf5_path, feature_shape, image_paths))
    saver_thread.start()

    for images_batch in tqdm(data_loader, desc="Extracting Features"):
        images_batch = images_batch.to(device)
        with torch.no_grad():
            features_batch = model(images_batch)
            if len(features_batch.shape) > 2:
                features_batch = features_batch.reshape(features_batch.size(0), -1)
        features_queue.put(features_batch.cpu().numpy())

    features_queue.put(None)  # Signal the worker to exit
    saver_thread.join()


# def extract_and_save_features_batchwise(image_paths, model, transform, device, batch_size, hdf5_path):
#     """
#     Extract features in batches and save them incrementally to an HDF5 file.
#     Args:
#         image_paths (list): List of image paths.
#         model (torch model): Model to use for feature extraction.
#         transform (torchvision.transforms): Transformations to be applied to the images.
#         device (torch.device): Device to run the model on.
#         batch_size (int): Number of images to process in a batch.
#         hdf5_path (str): Path to the HDF5 file where features will be saved.
#     """

#     dataset = CustomImageDataset(image_paths, transform=transform)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     with h5py.File(hdf5_path, 'w') as h5f:
#         # Create datasets to store image paths and features
#         h5f.create_dataset('image_paths', data=image_paths, dtype=h5py.string_dtype(), compression="gzip")
        
#         # Initialize feature_shape based on the model's output
#         dummy_input = torch.zeros((1, 3, 480, 480)).to(device)
#         with torch.no_grad():
#             dummy_output = model(dummy_input)
#         feature_shape = dummy_output.shape[1:]
#         features_dset = h5f.create_dataset('features', shape=(len(image_paths), *feature_shape), dtype='float32', compression="gzip")

#         # Batch-wise feature extraction and saving
#         for i, images_batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Extracting Features"):
#             images_batch = images_batch.to(device)
#             with torch.no_grad():
#                 features_batch = model(images_batch)
#                 # If the model returns a tensor with more than 2 dimensions, flatten it
#                 if len(features_batch.shape) > 2:
#                     features_batch = features_batch.reshape(features_batch.size(0), -1)

#             # Incrementally save features
#             start_idx = i * batch_size
#             end_idx = start_idx + features_batch.shape[0]
#             features_dset[start_idx:end_idx] = features_batch.cpu().numpy()


def extract_feature_single_image(image_path, model, transform, device):
    """
    Load an image from a file, apply transformations, and extract features using a specified model.
    Args:
        image_path (str): Path to the image file.
        model (torch model): Model to use for feature extraction.
        transform (torchvision.transforms): Transformations to be applied to the image.
        device (torch.device): Device to run the model on.
    Returns:
        torch.Tensor: Extracted features.
    """
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        model.eval()
        feature = model(image_tensor)

        # Flatten the feature if it's not a 1D vector
        if len(feature.shape) > 2:
            feature = feature.reshape(feature.size(0), -1)

    return feature.cpu()

# def extract_and_save_features_batchwise(image_paths, model, device, batch_size, hdf5_path):
#     """
#     Extract features in batches and save them incrementally to an HDF5 file.
#     """
#     transform = transforms.Compose([
#         transforms.Resize(512),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     dataset = CustomImageDataset(image_paths, transform=transform)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    
#     with h5py.File(hdf5_path, 'w') as h5f:
#         # Create datasets to store image paths and features
#         h5f.create_dataset('image_paths', data=image_paths, dtype=h5py.string_dtype(), compression="gzip")
#         feature_shape = (1920,)  # Adjust based on the model output
#         features_dset = h5f.create_dataset('features', shape=(len(image_paths), *feature_shape), dtype='float32', compression="gzip")

#         # Batch-wise feature extraction and saving
#         for i, images_batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Extracting Features"):
#             images_batch = images_batch.to(device)
#             with torch.no_grad():
#                 features_batch = model(images_batch)
#                 features_batch = features_batch.reshape(features_batch.size(0), -1)

#             # Incrementally save features
#             start_idx = i * batch_size
#             end_idx = start_idx + features_batch.shape[0]
#             features_dset[start_idx:end_idx] = features_batch.cpu().numpy()


# def extract_feature_single_image(image_path, model, device):
#     """
#     Load an image from a file, apply transformations, and extract features using a specified model.
#     """
#     # Define the transformations
#     transform = transforms.Compose([
#         transforms.Resize(512),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # Load and preprocess the image
#     image = Image.open(image_path).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0).to(device)

#     # Extract features
#     with torch.no_grad():
#         model.eval()
#         feature = model(image_tensor)
#         feature = feature.reshape(feature.size(0), -1)

#     return feature.cpu()


def image_query(query_image_feature, hdf5_path, k=10):
    """
    Query the dataset to find 'k' most similar images using their features.
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Load features and image paths from the HDF5 file
        features = f['features'][:]
        image_paths = f['image_paths'][:]
        
        # Fit the NearestNeighbors model
        neigh = NearestNeighbors(n_neighbors=k, metric='euclidean')
        neigh.fit(features)
        
        # Find the k-nearest neighbors
        distances, indices = neigh.kneighbors([query_image_feature.numpy().flatten()])
        nearest_image_paths = [image_paths[index] for index in indices.flatten()]

    return nearest_image_paths, distances.flatten()


def perform_multiple_retrievals(query_feature, hdf5_path, num_iterations=5, top_k=3):
    all_retrievals = []
    for _ in range(num_iterations):
        nearest_image_paths, _ = image_query(query_feature, hdf5_path=hdf5_path, k=top_k)
        all_retrievals.extend(nearest_image_paths)
    return all_retrievals


def score_and_rank_images(retrieved_images):
    image_scores = defaultdict(int)
    for img_path in retrieved_images:
        image_scores[img_path] += 1  
    return image_scores


def select_most_probable_image(image_scores):
    most_probable_image, score = max(image_scores.items(), key=lambda item: item[1])
    return most_probable_image, score


def load_image_paths_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]
    

def get_image_label_path(file_path):
    """
    Extract the label name along with its parent folder name from the file path.
    """
    # Ensure the file_path input is a string. If it's bytes, decode it.
    if isinstance(file_path, bytes):
        file_path = file_path.decode("utf-8")  # or use the appropriate encoding for your files

    # Get the full path of the parent directory
    parent_dir = os.path.dirname(file_path)

    # Find all .pickle files in the parent directory
    pickle_files = [f for f in os.listdir(parent_dir) if f.endswith('.pickle')]
    
    # Ensure there's exactly one .pickle file in the directory
    if len(pickle_files) != 1:
        raise ValueError(f"Expected exactly one .pickle file in the directory, but found {len(pickle_files)}")

    # Construct the full path to the .pickle file
    full_image_identifier = os.path.join(parent_dir, pickle_files[0])

    return full_image_identifier

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def quaternion_to_euler(q):
    w, x, y, z = q
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def direction_to_quaternion(direction):
    up = np.array([0.0, 1.0, 0.0])
    z_axis = -direction
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:  # Handle the case where up and z_axis are parallel
        x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)

    rot_matrix = np.array([normalize_vector(x_axis), normalize_vector(y_axis), normalize_vector(z_axis)]).T
    q = np.zeros(4)
    q[0] = np.sqrt(max(0, 1 + rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2])) / 2
    q[1] = np.sqrt(max(0, 1 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])) / 2
    q[2] = np.sqrt(max(0, 1 - rot_matrix[0, 0] + rot_matrix[1, 1] - rot_matrix[2, 2])) / 2
    q[3] = np.sqrt(max(0, 1 - rot_matrix[0, 0] - rot_matrix[1, 1] + rot_matrix[2, 2])) / 2
    q[1] *= np.sign(q[1] * (rot_matrix[2, 1] - rot_matrix[1, 2]))
    q[2] *= np.sign(q[2] * (rot_matrix[0, 2] - rot_matrix[2, 0]))
    q[3] *= np.sign(q[3] * (rot_matrix[1, 0] - rot_matrix[0, 1]))
    return q

def record_camera_orientation_and_distance(camera_coords, avatar_coords):
    camera_coords = np.array(camera_coords)
    avatar_coords = np.array(avatar_coords)

    # Calculate the distance between the camera and the avatar
    distance = np.linalg.norm(camera_coords - avatar_coords)

    # Calculate the direction vector and normalize it
    direction = normalize_vector(camera_coords - avatar_coords)

    # Calculate the quaternion from the direction vector
    quaternion = direction_to_quaternion(direction)

    # Convert the quaternion to Euler angles
    euler_angles = quaternion_to_euler(quaternion)

    # Save the distance and orientation
    record = {
        'distance': distance,
        'orientation': euler_angles
    }

    return record

def read_bone_capture_data(pickle_file_path, dataset_dir, avatars_dir, scenes_dir):
    with open(pickle_file_path, 'rb') as file:
        bone_capture_data = pickle.load(file)

    # Extract 'workers_name_list' and 'lighting' from the first level of the dictionary
    workers_name_list = bone_capture_data['workers_name_list']
    lighting = bone_capture_data['lighting']
    
    # Extract target_avatar
    target_avatar = []
    for name in workers_name_list:
        if name in pickle_file_path:
            target_avatar.append(name)
    if len(target_avatar) != 1:
        print('Error in target avatar')

    # Go into the "1" dict and read 'camera_location'
    camera_location = bone_capture_data['1'].get('camera_location', None)

    # Extract the root_bones for each worker and then find the bone location from the 'bone_location_3d' dict
    worker_bones_info = {}
    for worker_name in workers_name_list:
        worker_data = bone_capture_data['1'].get(worker_name, None)
        if worker_data:
            root_bones = worker_data.get('root_bones', [])
            bone_locations = {bone: worker_data['bone_location_3d'][bone] for bone in root_bones if bone in worker_data['bone_location_3d']}
            worker_bones_info[worker_name] = bone_locations

    # Extract the main part of the file path (the name of the image sequence folder)
    main_part = os.path.basename(os.path.dirname(pickle_file_path))  # cross-platform directory name extraction
    parts = main_part.split("_")[2:] # name of the folder without Random_2
    index_before_second_v = next(i for i, part in enumerate(parts) if part.startswith('V'))
    scene_name_parts = parts[:index_before_second_v + 1]
    scene_name = "_".join(scene_name_parts)
    camera_record = record_camera_orientation_and_distance(camera_location, bone_capture_data['1'][target_avatar[0]]['bone_location_3d'][bone_capture_data['1'][target_avatar[0]]['root_bones'][0]])
    
    # Return the extracted information in a dictionary
    return {
        'dataset_dir':  dataset_dir, 
        'avatars_dir': avatars_dir, 
        'scenes_dir': scenes_dir,
        'lighting': lighting,
        'camera_location': camera_location,
        'distance': float(camera_record['distance']),
        'orientation': [float(item) for item in camera_record['orientation']],
        # 'camera_orientation_distance': 
        'worker_bones_info': worker_bones_info,
        'scene_name': scene_name,
        'target_avatar': target_avatar[0]
    }


def create_config_yaml(bone_capture_output, output_path, extra_config):
    # Combine the bone capture data with the extra configuration
    combined_data = {**bone_capture_output, **extra_config}
    
    # Write the combined data to the YAML file
    with open(output_path, 'w') as file:
        yaml.dump(combined_data, file, default_flow_style=False)
    
    return output_path


def pram_retreival(query_image_path, model, transform, device, hdf5_path,  dataset_dir, avatars_dir, scenes_dir, num_iterations=3, top_k=3):
    # Extract features of the query image
    query_feature = extract_feature_single_image(query_image_path, model, transform, device)

    # Perform the query multiple times and score the results
    retrieved_images = perform_multiple_retrievals(query_feature, hdf5_path=hdf5_path, num_iterations=num_iterations, top_k=top_k)
    image_scores = score_and_rank_images(retrieved_images)

    # Select the most probable image and its score
    most_probable, score = select_most_probable_image(image_scores)
    
    if isinstance(most_probable, bytes):
        most_probable = most_probable.decode('utf-8')
    
    most_probable = most_probable.replace('\\', '/')
    
    # If HDF5 was built on a different OS, translate the stored paths.
    # Set SDAL_HDF5_PATH_PREFIX and SDAL_HDF5_PATH_REPLACE env vars to remap, e.g.:
    #   export SDAL_HDF5_PATH_PREFIX="D:/SyntheticData_SDAL_Features"
    #   export SDAL_HDF5_PATH_REPLACE="/mnt/data/SyntheticData_SDAL_Features"
    prefix = os.environ.get('SDAL_HDF5_PATH_PREFIX')
    replace = os.environ.get('SDAL_HDF5_PATH_REPLACE')
    if prefix and replace and most_probable.startswith(prefix):
        most_probable = most_probable.replace(prefix, replace)
    #########################################################
    # Extract and print the image name
    most_probable_image_label_path = get_image_label_path(most_probable)
    most_probable_image_info = read_bone_capture_data(most_probable_image_label_path, dataset_dir, avatars_dir, scenes_dir)
    
    return most_probable, most_probable_image_info, most_probable_image_label_path


def create_data_gen_env(
        query_image_path, 
        data_gen_env_dir, 
        model, 
        transform, 
        device, 
        hdf5_path, 
        dataset_dir, 
        avatars_dir, 
        scenes_dir,
        scene_collection_dir,
        image_size=224,
        framerate = 50, 
        top_k=3,
        decomposer_iterations=3,
        synth_generation_premutation=3,
        num_containers=3,
        run_root: Path = None,
        config_name: str = "config.json",
        blender_bin: str = "blender",
        visualize=False, 
        default_config_dir='sdal_utils/Data_Generator/default_config.yaml', 
        track=True,
        logger=None
    ):
    """
    Create a data generation environment based on a query image. This function 
    sets up an environment for generating synthetic data using a pre-trained model 
    and specific configurations. It also provides options for visualization and tracking.
    """

    # Set the default configuration if not provided   
    if default_config_dir is None:
        default_config = { 
            'Threshold': 0.001,
            'max_bounces': 4,
            'samples': 1024,
            'tile_size': 256,
            'resolution_x': int(image_size),
            'resolution_y': int(image_size),
        }
    elif isinstance(default_config_dir, str):
        with open(default_config_dir, 'r') as file:
            default_config = yaml.load(file, Loader=yaml.FullLoader)

    default_config['Number_of_Image_Sequences'] = synth_generation_premutation
    default_config['Framerate'] = framerate

    # Extract image and folder names from the query path
    image_name, folder_name = os.path.basename(query_image_path).replace('.jpg', ''), os.path.basename(os.path.dirname(query_image_path))

    # Perform parameter retrieval
    logger.info(f'Retrieving similar image for {str(query_image_path)}...')
    most_probable, most_probable_image_info, most_probable_image_label_path = pram_retreival(query_image_path, model, transform, device, hdf5_path, dataset_dir, str(avatars_dir), str(scenes_dir), num_iterations=decomposer_iterations, top_k=top_k)

    found_img = os.path.basename(os.path.dirname(most_probable)).__str__().strip('b').strip("'")
    logger.info(f'Similar image found: {found_img}')
    logger.info(f"Target worker: {most_probable_image_info['target_avatar']}")
    logger.info(f"Number of workers: {len(most_probable_image_info['worker_bones_info'])}")

    # Checking the retrieved data
    logger.info('Checking the retrieved data with avatar and scene repositories...')
    available_avatars = [avatar.strip('.blend') for avatar in os.listdir(avatars_dir)]
    available_scenes = [scene.strip('.blend') for scene in os.listdir(scenes_dir)]
    # Change avatars with what we have
    avatars_in_scene = list(most_probable_image_info['worker_bones_info'].keys()) # Create a list of keys to iterate over
    for worker in avatars_in_scene:
        if worker not in available_avatars:
            # Find a matching avatar
            selected_candidate = find_matching_avatar([worker], available_avatars)
            if selected_candidate:
                most_probable_image_info['worker_bones_info'][selected_candidate] = most_probable_image_info['worker_bones_info'].pop(worker)
                logger.info(f'{worker} not found in avatars dir. Changing {worker} to {selected_candidate}')
            else:
                logger.warning(f'{worker} not found in avatars dir. No similar avatar found. Assigning randomly.')
                selected_candidate = random.choice(available_avatars)
                most_probable_image_info['worker_bones_info'][selected_candidate] = most_probable_image_info['worker_bones_info'].pop(worker)
                logger.info(f'Changing {worker} to {selected_candidate}')
            
            if worker in most_probable_image_info['target_avatar']:
                most_probable_image_info['target_avatar'] = selected_candidate

    # Change scene with what we have
    if most_probable_image_info['scene_name'] not in available_scenes:
        selected_candidate = random.choice(available_scenes)
        logger.info(f"{most_probable_image_info['scene_name']} not found in scene dir. Changing {most_probable_image_info['scene_name']} to {selected_candidate}")
        most_probable_image_info['scene_name'] = selected_candidate

    if most_probable_image_info is None:
        logger.error("No probable image information found.")
        raise ValueError

    logger.info('Retrieved data checked successfully.')
    # Save configuration in run-scoped folder (preferred) or legacy location
    # - Docker-free pipeline: write JSON config into run_root
    # - Legacy pipeline: write YAML config into data_gen_env_dir/config.yaml
    default_config_with_containers = {**default_config, 'num_containers': num_containers}

    if run_root is not None:
        run_root = Path(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        config_path = run_root / config_name
        combined_data = {**most_probable_image_info, **default_config_with_containers}
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=2)
        if logger:
            logger.info(f"Wrote generator config JSON: {config_path}")

        # Normalize the avatar location (writes back into config JSON)
        normalized = normalize_all_location(
            scene_collection_dir=scene_collection_dir,
            config_file=config_path,
            input_json=run_root / "input_points.json",
            output_json=run_root / "output_vectors.json",
            blender_bin=blender_bin,
        )
    else:
        output_config_yaml_path = os.path.join(data_gen_env_dir, 'config.yaml')
        _ = create_config_yaml(most_probable_image_info, output_config_yaml_path, default_config_with_containers)

        normalized = normalize_all_location(
            scene_collection_dir=scene_collection_dir,
            config_file=output_config_yaml_path,
            input_json='input_points.json',
            output_json='output_vectors.json',
            blender_bin=blender_bin,
        )
    if normalized:
        logger.info('Normalization successful')

    # Visualization block
    if visualize:
        query_image = Image.open(query_image_path)
        most_probable_img = Image.open(most_probable)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(query_image)
        axes[0].set_title("Query Image")
        axes[0].axis('off')
        axes[1].imshow(most_probable_img)
        axes[1].set_title("Most Probable Outcome")
        axes[1].axis('off')

        plt.savefig(f"{data_gen_env_dir}/comparison_plot_{image_name}.png")

    # Tracking block
    if track:
        track_csv_path = './sdal_utils/Data_Generator/query_retrieved_imgs.csv'
        file_exists = os.path.isfile(track_csv_path)
        with open(track_csv_path, mode='a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Query Image Path', 'Most Probable Synthetic Image Path'])
            writer.writerow([query_image_path, most_probable])

def get_uncertain_images(data_yaml, model_weights, img_num=10, imgsz=960, confidence_based=False):
    # Calculate loss for val set
    if not confidence_based:
        mp, mr, map50, map, loss, maps, t, image_losses_dict = test(
            data_yaml,
            weights=model_weights,
            batch_size=1,
            imgsz=imgsz,
            conf_thres=0.001,
            iou_thres=0.6,  # for NMS
            save_json=False,
            single_cls=False,
            augment=False,
            verbose=False,
            model=None,
            dataloader=None,
            # save_dir=Path(''),  # for saving images
            save_txt=False,  # for auto-labelling
            save_hybrid=False,  # for hybrid auto-labelling
            save_conf=False,  # save auto-label confidences
            plots=True,
            wandb_logger=None,
            # compute_loss=None,
            half_precision=True,
            trace=False,
            is_coco=False,
            v5_metric=False,
            training=True,
            device='0',
            task='val'
            )
        sorted_image_losses_tpl = sorted(image_losses_dict.items(), key=lambda x: x[1], reverse=True)
        return mp, mr, map50, map, loss, maps, t, sorted_image_losses_tpl[:img_num]
    else:
        mp, mr, map50, map, loss, maps, t, image_losses_dict, uc_dict = test(
            data_yaml,
            weights=model_weights,
            confidence_based=True,
            batch_size=1,
            imgsz=imgsz,
            conf_thres=0.001,
            iou_thres=0.6,  # for NMS
            save_json=False,
            single_cls=False,
            augment=False,
            verbose=False,
            model=None,
            dataloader=None,
            # save_dir=Path(''),  # for saving images
            save_txt=False,  # for auto-labelling
            save_hybrid=False,  # for hybrid auto-labelling
            save_conf=False,  # save auto-label confidences
            plots=True,
            wandb_logger=None,
            # compute_loss=None,
            half_precision=True,
            trace=False,
            is_coco=False,
            v5_metric=False,
            training=True,
            device='0',
            task='val'
            )
        sorted_uncertaintiy_tpl = sorted(uc_dict.items(), key=lambda x: x[1], reverse=True)
        return mp, mr, map50, map, loss, maps, t, sorted_uncertaintiy_tpl[:img_num]
    

def merge_images(failure_case_image_path, generated_images_paths):
    # Load the failure case image
    failure_case_image = Image.open(failure_case_image_path).convert("RGB")
    
    width, height = failure_case_image.size

    # Load the generated images
    generated_images = [Image.open(img_path).convert("RGB").resize(failure_case_image.size) for img_path in generated_images_paths]
    
    # Define the size for the combined image
    total_width = width * (len(generated_images) + 1)
    combined_image = Image.new('RGB', (total_width, height))
    
    # Paste the failure case image
    combined_image.paste(failure_case_image, (0, 0))

    # Paste the generated images
    for i, img in enumerate(generated_images):
        combined_image.paste(img, ((i + 1) * width, 0))

    return combined_image

def clean_stored_cache(data_yaml):
    with open(data_yaml) as f:
        yaml_info = yaml.load(f, Loader=yaml.FullLoader)
    train_data_path = Path(yaml_info['train'])
    val_data_path = Path(yaml_info['val'])
    test_data_path = Path(yaml_info['test'])

    for data_path in [train_data_path, val_data_path, test_data_path]:
        for file in os.listdir(data_path):
            if file.endswith('.cache'):
                os.remove(os.path.join(data_path, file))

def get_new_weights(result, keywords, warmed_up_model_weights, PROJECT_DIR):
    weights = warmed_up_model_weights
    if all([keyword in result.stdout for keyword in keywords]):
        experiments_folders = [file for file in os.listdir(PROJECT_DIR)]
        try:
            # weights_path = re.search(r'(yolov\d+\\runs\\train\\[^\\]+\\weights\\last)', result.stdout).group(1).split('\\') # ['yolov7', 'runs', 'train', 'exp*', 'weights', 'last']
            weights_path = re.search(r'(yolov\d+/runs/train/[^/]+/weights/last)', result.stdout).group(1).split('/')
            probable_weights_folder = Path(PROJECT_DIR) / weights_path[-3] / weights_path[-2] / weights_path[-1]
        except:
            probable_weights_folder = None
    elif len(experiments_folders) > 0:
        last_training_folder = sorted([int(file.strip('exp')) if file.strip('exp').isdigit() else 0 for file in experiments_folders])[-1]
        weights = Path(PROJECT_DIR) / last_training_folder / 'weights' / 'best.pt'
        if not weights.exists() and probable_weights_folder and probable_weights_folder.exists():
            weights = probable_weights_folder
    else:
        print('No new weights found. Using warmed up model weights.')
    return weights

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)