import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.models import densenet201
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import subprocess
from sdal_utils.sdal_utils import create_data_gen_env
from sdal_utils.pickle2yolo import pickle2yolo
import shutil
from pathlib import Path

def decomposer(
        query_image, 
        scene_collection_dir, 
        avatars_dir, 
        scenes_dir,
        image_size=224, 
        top_k=3, 
        decomposer_iterations=3, 
        synth_generation_premutation=3, 
        generation_framerate=50, 
        data_gen_env_dir='./sdal_utils/Data_Generator', 
        hdf5_path='./features/features_DenseNet201.hdf5', 
        dataset_dir='./sdal_utils/Data_Generator/Dataset', 
        logger=None
        ) -> bool:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'BiT' in str(hdf5_path):
        model = timm.create_model('resnetv2_152x4_bitm', pretrained=True)
        model.eval()
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        model.reset_classifier(0)
        model = model.to(device)
    elif 'DenseNet' in str(hdf5_path):
        model = densenet201(pretrained=True)
        model.classifier = nn.Identity()
        model = model.to(device)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        if logger:
            logger.error("Unsupported model type for feature extraction.")
        raise NotImplementedError("Unsupported model type for feature extraction.")

    create_data_gen_env(
        query_image_path=query_image, 
        data_gen_env_dir=data_gen_env_dir, 
        model=model, 
        transform=transform, 
        image_size=image_size,
        device=device, 
        hdf5_path=hdf5_path, 
        dataset_dir=dataset_dir, 
        avatars_dir=avatars_dir, 
        scenes_dir=scenes_dir, 
        scene_collection_dir=scene_collection_dir, 
        framerate=generation_framerate, 
        top_k=top_k, 
        decomposer_iterations=decomposer_iterations, 
        synth_generation_premutation=synth_generation_premutation, 
        logger=logger)
    
    return True

def generate_data(
        # data_gen_env_dir='./sdal_utils/Data_Generator', 
        # empty_dir='./sdal_utils/Data_Generator/Empty.blend', 
        # code_dir='./sdal_utils/Data_Generator/231109_Adaptive_Data_Generator.py',
        docker_run_script,
        logger=None
        ):
    if logger:
        logger.info('Data generation ...')
    try:
        # result = subprocess.run(["/bin/bash", docker_run_script], check=True)
        result = subprocess.run(['sudo', str(docker_run_script)], check=True)
        logger.debug(f"Data generation engine output: {result.stdout}")
    except:
        if logger: logger.error(f"Error running data generation engine")
    return result.returncode==0

def generate_yolo_labels(input_folders, output_path, save_txt, logger=None):
    labels = []
    example_images = {}
    for input_folder in input_folders:
        try:
            if not os.path.exists(os.path.join(output_path, 'labels')):
                os.makedirs(os.path.join(output_path, 'labels'))
            label = pickle2yolo(input_folder, os.path.join(output_path, 'labels'), save_txt=save_txt)
            logger.info(f"Generated {len(label.values())} number of YOLO labels for {input_folder}")
            labels.append(label)
        except Exception as e:
            if logger:
                logger.error(f"Error generating YOLO labels for {input_folder}: {str(e)}")
            continue
        
        logger.info(f"Copying {len([file for file in os.listdir(input_folder) if 'jpg' in file])} number of images to {output_path}")
        for img in os.listdir(input_folder):
            if '.jpg' not in img:
                continue
            if not os.path.exists(os.path.join(output_path, 'images')):
                os.makedirs(os.path.join(output_path, 'images'))
            shutil.copy(input_folder / img, output_path / 'images' / f"{os.path.basename(input_folder)}_{img}")

            if 'test0001' not in img and example_images.get(input_folder) is None:
                example_images[input_folder] = output_path / 'images' / f"{os.path.basename(input_folder)}_{img}"

    return labels, example_images

def oracle(
        query_image_path, 
        output_path, 
        avatars_dir, 
        scenes_dir, 
        scene_collection_dir,
        image_size=224, 
        top_k=3, 
        decomposer_iterations=3, 
        synth_generation_premutation=3, 
        generation_framerate=50,
        hdf5_path='./features/features_DenseNet201.hdf5', 
        data_gen_env_dir='./sdal_utils/Data_Generator', 
        save_yolo_labels=True, 
        dataset_used_dir='./sdal_utils/Data_Generator/Dataset_used', 
        data_gen_docker_script='run_blendcon.sh',
        logger=None,
        ) -> tuple:
    
    data_gen_env_dir = Path(data_gen_env_dir)
    output_path = Path(output_path)
    dataset_used_dir = Path(dataset_used_dir)

    if not dataset_used_dir.exists():
        os.makedirs(dataset_used_dir)

    try:
        decomposed = decomposer(
            query_image=query_image_path, 
            data_gen_env_dir=data_gen_env_dir, 
            hdf5_path=hdf5_path, 
            generation_framerate=generation_framerate, 
            top_k=top_k,
            image_size=image_size,
            decomposer_iterations=decomposer_iterations, 
            synth_generation_premutation=synth_generation_premutation, 
            avatars_dir=avatars_dir, 
            scenes_dir=scenes_dir, 
            scene_collection_dir=scene_collection_dir, 
            logger=logger
            )
        
        if decomposed:
            result = generate_data(docker_run_script=data_gen_env_dir / data_gen_docker_script, logger=logger)
        else:
            if logger:
                logger.error("Decomposition failed.")
            raise ValueError('Decomposition failed')
        
        if result:
            if logger:
                logger.info('Data generation was successful')
        else:
            if logger:
                logger.error(f"Data generation failed: {result.stdout}\n{result.stderr}")
            raise ValueError(f'Data generation failed. Unable to run blendcon docker with {data_gen_env_dir / data_gen_docker_script}')

        # Process data from all Dataset directories (Dataset_1, Dataset_2, Dataset_3) and add prefix
        input_folders = []
        for idx in range(1, 4):
            dataset_dir = data_gen_env_dir / f'Dataset_{idx}'
            for folder in os.listdir(dataset_dir):
                folder_path = dataset_dir / folder
                if os.path.isdir(folder_path):
                    # Add a prefix to the folder name to avoid conflicts
                    new_folder_name = f'D_{idx}_{folder}'
                    new_folder_path = dataset_dir / new_folder_name
                    os.rename(folder_path, new_folder_path)  # Rename the folder with the new prefix
                    input_folders.append(new_folder_path)

        logger.info(f"Generating YOLO labels for {input_folders} and moving to {output_path}")
        labels, example_images = generate_yolo_labels(input_folders, output_path, save_yolo_labels, logger=logger)
        logger.info(f'Removing generated data to archive: {dataset_used_dir}')
        for item_path in input_folders:
            if os.path.exists(item_path):
                dst = dataset_used_dir / item_path.name
                exist = os.path.exists(dst)
                iteration = 0
                while exist:
                    iteration += 1
                    dst = dataset_used_dir / f"{item_path.name}-{iteration}"
                    exist = os.path.exists(dst)
                shutil.move(item_path, dst)
            else:
                if logger:
                    logger.error(f"{item_path} not found")
                continue

        return labels, example_images

    except Exception as e:
        if logger:
            logger.error(f"Error in oracle function: {str(e)}")
            logger.info(f'Removing incomplete generated data to archive: {dataset_used_dir}')
        for item_path in input_folders:
            if os.path.exists(item_path):
                dst = dataset_used_dir / item_path.name
                exist = os.path.exists(dst)
                iteration = 0
                while exist:
                    iteration += 1
                    dst = dataset_used_dir / f"{item_path.name}-{iteration}"
                    exist = os.path.exists(dst)
                shutil.move(item_path, dst)
            else:
                if logger:
                    logger.error(f"{item_path} not found")
                continue
        raise


if __name__ == "__main__":
    # Example usage - replace with your actual paths
    query_image_path = r"./datasets/target/test/images/example.jpg"
    output_path = r'./sdal_utils/Data_Generator/yolo_dataset'

    logger = logging.getLogger('oracle_logger')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    labels = oracle(query_image_path, output_path, logger=logger)
