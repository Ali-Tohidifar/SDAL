import logging
import sys
import os
import time
import yaml
from pathlib import Path
import subprocess
import warnings
import argparse
from datetime import datetime
import dotenv
import torch
import torchvision.transforms as transforms
from PIL import Image
import random

dotenv.load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

# Add YOLOv7 directory to Python path
sys.path.append('./yolov7')

from synthetic_oracle import oracle
from sdal_utils.sdal_utils import get_uncertain_images, merge_images, clean_stored_cache, get_new_weights, load_yaml
warnings.filterwarnings("ignore")

# project dir for yolo training
PROJECT_DIR = './yolov7/runs/train'


def train_yolo_sdal(paths, hyp, weights, project_dir, log_dir, logger, cycle, mode="sequential", checkpoint=None, save_checkpoint_func=None):
    """
    Trains YOLO models with two modes:
    1. Sequentially: Train first with synthetic data and then with real data.
    2. Combined: Combine synthetic and real data and train together.

    :param paths: Dictionary of paths for configuration and data.
    :param hyp: Dictionary of hyperparameters.
    :param weights: Initial weights for training.
    :param project_dir: Directory where YOLOv7 training runs are saved.
    :param log_dir: Directory for logs.
    :param logger: Logger instance.
    :param cycle: The current SDAL cycle number.
    :param mode: Training mode, either "sequential" or "combined".
    :param checkpoint: Checkpoint dictionary to track progress.
    :param save_checkpoint_func: Function to save checkpoint data.
    :return: Final trained weights path.
    """

    # Clean cache for training
    clean_stored_cache(paths['synth_data_yaml'])
    clean_stored_cache(paths['real_data_yaml'])

    # Use python from the activated conda environment
    # If you need a specific python path, modify this variable
    PYTHON_PATH="python"  # or specify: "/path/to/conda/envs/sdal/bin/python"

    if mode == "sequential":
        # Sequential Mode
        logger.info(f"Training mode: Sequential for cycle {cycle}")

        # Step 3: Train on Synthetic Data
        timestamp = datetime.now().strftime('%y-%m-%d-%H-%M')
        yolo_synth_name = f'{timestamp}_cycle_{cycle}_synth'
        # Remove resume flag - always start training from beginning
        train_prompt = f"{PYTHON_PATH} yolov7/train.py --img-size {int(hyp['image_size'])} --batch 128 --epochs {hyp['synth_epcohs']} --data {paths['synth_data_yaml']} --weights {weights} --single-cls --project {project_dir} --name {yolo_synth_name} --device {hyp['device']}"

        logger.info(f"Starting synthetic data training for cycle {cycle}: {train_prompt}")
        try:
            result = subprocess.run(train_prompt, shell=True, capture_output=True, text=True, check=True, encoding='utf-8', env=os.environ.copy())
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with return code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            checkpoint["synth_interrupted"] = True
            save_checkpoint_func(checkpoint)
            logger.error(f"Error in synthetic data training for cycle {cycle}.")
            return weights

        # Check if training produced new weights
        if os.path.exists(f'{project_dir}/{yolo_synth_name}/weights/best.pt'):
            weights = f'{project_dir}/{yolo_synth_name}/weights/best.pt'
        else:
            warnings.warn('Synthetic training did not produce any weights. Using previous weights.')
            logger.error('Synthetic training did not produce any weights. Using previous weights.')

        checkpoint["last_synth_weights"] = weights
        checkpoint["synth_interrupted"] = False  # Reset interruption flag after successful training
        save_checkpoint_func(checkpoint)
        logger.info(f"Completed synthetic data training for cycle {cycle}.")

        # Step 4: Train on Real Data
        timestamp = datetime.now().strftime('%y-%m-%d-%H-%M')
        yolo_real_name = f'{timestamp}_cycle_{cycle}_real'
        # Remove resume flag - always start training from beginning
        train_prompt = f"{PYTHON_PATH} yolov7/train.py --img-size {int(hyp['image_size'])} --batch 128 --epochs {hyp['real_epochs']} --data {paths['real_data_yaml']} --weights {weights} --single-cls --project {project_dir} --name {yolo_real_name} --device {hyp['device']}"

        logger.info(f"Starting real data training for cycle {cycle}: {train_prompt}")
        try:
            result = subprocess.run(train_prompt, shell=True, capture_output=True, text=True, check=True, encoding='utf-8', env=os.environ.copy())
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with return code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            checkpoint["real_interrupted"] = True
            save_checkpoint_func(checkpoint)
            logger.error(f"Error in real data training for cycle {cycle}.")
            return weights

        if os.path.exists(f'{project_dir}/{yolo_real_name}/weights/best.pt'):
            weights = f'{project_dir}/{yolo_real_name}/weights/best.pt'
        else:
            warnings.warn('Real training did not produce any weights. Using previous weights.')
            logger.error('Real training did not produce any weights. Using previous weights.')

        checkpoint["last_real_weights"] = weights
        checkpoint["real_interrupted"] = False  # Reset interruption flag after successful training
        save_checkpoint_func(checkpoint)
        logger.info(f"Completed real data training for cycle {cycle}.")

    elif mode == "combined":
        # Combined Mode
        logger.info(f"Training mode: Combined for cycle {cycle}")

        # Combine datasets in a temporary config
        combined_data_yaml = "./data_cfg/combined_data.yaml"
        with open(combined_data_yaml, 'w') as file:
            yaml.dump({
                'train': [load_yaml(paths['synth_data_yaml'])['train'], load_yaml(paths['real_data_yaml'])['train']],
                'val': load_yaml(paths['real_data_yaml'])['val'],
                'test': load_yaml(paths['real_data_yaml'])['test'],
                'nc': 1,  # number of classes
                'names': ['worker'],  # class names
            }, file)

        total_epochs = hyp['synth_epcohs'] + hyp['real_epochs']
        timestamp = datetime.now().strftime('%y-%m-%d-%H-%M')
        yolo_combined_name = f'{timestamp}_cycle_{cycle}_combined'
        # Remove resume flag - always start training from beginning
        train_prompt = f"{PYTHON_PATH} yolov7/train.py --img-size {int(hyp['image_size'])} --batch 128 --epochs {total_epochs} --data {combined_data_yaml} --weights {weights} --single-cls --project {project_dir} --name {yolo_combined_name} --device {hyp['device']}"

        logger.info(f"Starting combined data training for cycle {cycle}: {train_prompt}")
        try:
            result = subprocess.run(train_prompt, shell=True, capture_output=True, text=True, check=True, encoding='utf-8', env=os.environ.copy())
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with return code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            checkpoint["combined_interrupted"] = True
            save_checkpoint_func(checkpoint)
            logger.error(f"Error in combined data training for cycle {cycle}.")
            return weights

        if os.path.exists(f'{project_dir}/{yolo_combined_name}/weights/best.pt'):
            weights = f'{project_dir}/{yolo_combined_name}/weights/best.pt'
        else:
            warnings.warn('Combined training did not produce any weights. Using previous weights.')
            logger.error('Combined training did not produce any weights. Using previous weights.')

        checkpoint["last_combined_weights"] = weights
        checkpoint["combined_interrupted"] = False  # Reset interruption flag after successful training
        save_checkpoint_func(checkpoint)
        logger.info(f"Completed combined data training for cycle {cycle}.")

    else:
        logger.error(f"Invalid mode: {mode}. Choose 'sequential' or 'combined'.")
        raise ValueError("Invalid training mode.")

    return weights

def get_random_images(data_yaml, img_num=10):
    """
    Get random images from the validation set.
    return a list of tuples (img_path, dummy_loss)
    """
    
    with open(data_yaml, 'r') as file:
        data_yaml = yaml.safe_load(file)
    val_data_path = data_yaml['val']
    val_imgs = os.listdir(os.path.join(val_data_path, 'images'))
    
    # Ensure we don't try to sample more images than available
    img_num = min(img_num, len(val_imgs))
    random_images = random.sample(val_imgs, img_num)
    
    # Process the random images - just get the names without resizing
    processed_images = []
    for img_path in random_images:
        try:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            processed_images.append({
                'name': img_name,
                'path': os.path.join(val_data_path, 'images', img_path)
            })
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    # Return list of tuples (image_name, dummy_loss)
    return [(item['name'], 0) for item in processed_images]



def SDAL(config):
    # Set environment variable for subprocess runs
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # Extract paths from config
    paths = config['paths']
    hyp = config['hyp']
    exp_name = config['exp_name']
    resume = config.get('resume', False)

    # First, find the most recent log directory to load checkpoint from
    log_base_dir = paths['log_dir']
    most_recent_checkpoint = None
    most_recent_weights = None
    
    # Find the most recent log directory with a checkpoint, but only if resume is True
    if resume and os.path.exists(log_base_dir):
        log_dirs = [d for d in os.listdir(log_base_dir) if os.path.isdir(os.path.join(log_base_dir, d)) and d.endswith(f"SDAL_{os.path.basename(args.hyp).split('.')[0]}")]
        if log_dirs:
            # Sort by timestamp (most recent last)
            log_dirs.sort()
            most_recent_log = os.path.join(log_base_dir, log_dirs[-1])
            most_recent_checkpoint_path = os.path.join(most_recent_log, "checkpoint.yaml")
            
            if os.path.exists(most_recent_checkpoint_path):
                with open(most_recent_checkpoint_path, "r") as file:
                    most_recent_checkpoint = yaml.safe_load(file)
                print(f"Found previous checkpoint in {most_recent_log}")
                
                # Check for the most recent weights
                if most_recent_checkpoint:
                    for weight_key in ["last_combined_weights", "last_real_weights", "last_synth_weights"]:
                        if weight_key in most_recent_checkpoint and os.path.exists(most_recent_checkpoint[weight_key]):
                            most_recent_weights = most_recent_checkpoint[weight_key]
                            print(f"Found previous weights: {most_recent_weights}")
                            break
    elif not resume:
        print("Resume flag not set - starting fresh without loading previous checkpoints")

    # Now create a new log directory for the current run
    log_dir = f"{log_base_dir}/{exp_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, filename=f'{log_dir}/logger.log', format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('SDAL')

    # Create a checkpoint file to track progress
    checkpoint_file = Path(log_dir) / "checkpoint.yaml"

    def load_checkpoint():
        # If we found a previous checkpoint and resume is True, use it; otherwise create a new one
        if resume and most_recent_checkpoint:
            logger.info("Resuming from previous checkpoint")
            return most_recent_checkpoint
        if resume and checkpoint_file.exists():
            logger.info("Resuming from existing checkpoint file")
            with open(checkpoint_file, "r") as file:
                return yaml.safe_load(file)
        logger.info("Starting with a fresh checkpoint")
        return {}

    def save_checkpoint(data):
        with open(checkpoint_file, "w") as file:
            yaml.safe_dump(data, file)

    checkpoint = load_checkpoint()
    
    # Use the most recent weights if available, otherwise use the default
    weights = most_recent_weights if most_recent_weights else paths['warmed_up_model_weights']
    logger.info(f"Starting with weights: {weights}")

    # Read val and train data path from yaml file
    real_val_data_path = Path(load_yaml(paths['real_data_yaml'])['val'])
    train_data_path = Path(load_yaml(paths['real_data_yaml'])['train'])

    synth_train_data_path = Path(load_yaml(paths['synth_data_yaml'])['train'])

    start_time = time.time()

    for i in range(hyp['SDAL_cycle']):
        cycle_start_time = time.time()
        logger.info(f'Starting Cycle {i + 1}')

        # Modified logic: Don't skip a cycle if it was interrupted
        cycle_interrupted = False
        if config.get('mode', 'sequential') == 'sequential':
            cycle_interrupted = checkpoint.get("synth_interrupted", False) or checkpoint.get("real_interrupted", False)
        else:  # combined mode
            cycle_interrupted = checkpoint.get("combined_interrupted", False)
            
        if checkpoint.get("current_cycle", 0) > i and not cycle_interrupted:
            logger.info(f"Skipping cycle {i + 1} as it has already been completed.")
            continue
        elif checkpoint.get("current_cycle", 0) == i+1 and cycle_interrupted:
            logger.info(f"Resuming cycle {i + 1} which was interrupted in previous run.")
            # Reset the current_cycle to force re-running this cycle
            checkpoint["current_cycle"] = i
            save_checkpoint(checkpoint)
        
        # # Initialize processed images if not present
        # if "processed_images" not in checkpoint:
        #     checkpoint["processed_images"] = {}
        #     save_checkpoint(checkpoint)
            
        # Initialize cycle-specific tracking
        if f"cycle_{i+1}_processed_images" not in checkpoint:
            checkpoint[f"cycle_{i+1}_processed_images"] = {}
            save_checkpoint(checkpoint)
        
        # using random selection from the real data
        if hyp['random_selection']:
            logger.info(f"Using random selection from the real data for cycle {i + 1}...")
            # Step 1: Get failure cases and augment training data
            logger.info(f"Getting uncertain images for cycle {i + 1}...")
            logger.info(f"Using {hyp['confidence_based']} confidence based approach for uncertainty calculation")
            failure_cases = get_random_images(paths['real_data_yaml'], img_num=hyp['failure_cases_to_generate'])
            logger.info(f"Failure cases for cycle {i + 1}: {failure_cases}")
            # import ipdb;ipdb.set_trace()
        else:
            

            # Step 1: Get failure cases and augment training data
            logger.info(f"Getting uncertain images for cycle {i + 1}...")
            logger.info(f"Using {hyp['confidence_based']} confidence based approach for uncertainty calculation")
            mp, mr, map50, map, loss, maps, t, failure_cases = get_uncertain_images(paths['real_data_yaml'], weights, img_num=hyp['failure_cases_to_generate'], imgsz=416, confidence_based=hyp['confidence_based'])
            logger.info({
                'Mean Precision': mp,
                'Mean Recall': mr,
                'Mean Average Precision @50': map50,
                'Mean Average Precision': map,
                'Validation Loss box': loss[0],
                'Validation Loss obj': loss[1],
                'Validation Loss cls': loss[2],
                "Time for Validation Prediction": t[0],
                "Time for Validation NMS": t[1],
                "Time for Validation Total": t[2],
                "Image Size yolo_test": t[3],
                "Batch Size yolo_test": t[5],
            })

        logger.info(f"Failure cases for cycle {i + 1}: {failure_cases}")
        total_img_num = 0
        total_label_num = 0

        # Step 2: Synthetic Data Generation
        for img, loss in failure_cases:
            img_path = img + '.jpg'
            informative_image_path = real_val_data_path / 'images' / img_path

            if checkpoint.get(f"cycle_{i+1}_processed_images", {}).get(img_path, False):
                logger.info(f"Skipping synthetic data generation for {img_path} in cycle {i+1} as it has already been processed.")
                continue

            try:
                logger.info(f"Passing image {img_path} to oracle for synthetic data generation...")
                labels, example_images = oracle(
                    informative_image_path,
                    output_path=synth_train_data_path,
                    image_size=int(hyp['image_size']),
                    hdf5_path=paths['hdf5_path'],
                    top_k=hyp['top_k'],
                    decomposer_iterations=hyp['decomposer_iterations'],
                    synth_generation_premutation=hyp['synth_generation_premutation'],
                    save_yolo_labels=True,
                    generation_framerate=hyp['generation_framerate'],
                    data_gen_env_dir=paths['data_gen_env_dir'],
                    dataset_used_dir=paths['dataset_used_dir'],
                    avatars_dir=paths['avatar_dir'],
                    scenes_dir=paths['scene_dir'],
                    scene_collection_dir=paths['old_scene_collection_dir'],
                    logger=logger
                )
                num_generated_images = [len(label.keys()) for label in labels]
                num_generated_labels = [sum(len(value_list) for value_list in data_dict.values()) for data_dict in labels]
                total_img_num += sum(num_generated_images)
                total_label_num += sum(num_generated_labels)

                generated_images_paths = [str(gen_img_dir) for folder, gen_img_dir in example_images.items()]
                merged_image = merge_images(str(informative_image_path), generated_images_paths)
                
                # Save the merged image to disk
                merged_image_path = f"./{log_dir}/cycle_{i + 1}_failure_case_{os.path.basename(img_path)}.jpg"
                merged_image.save(merged_image_path)
                
                # Log the name of the saved image
                logger.info(f"Saved Merged Image for Cycle {i + 1}: {merged_image_path}")

                # Update checkpoint
                checkpoint.setdefault("processed_images", {})[img_path] = True
                checkpoint.setdefault(f"cycle_{i+1}_processed_images", {})[img_path] = True
                save_checkpoint(checkpoint)
            except Exception as e:
                logger.error(f'Synthetic data generation Error: {str(e)}')
                print(f"Error in generating synthetic data for image {img_path}")
                continue

        logger.info({
            'Total Generated Images': total_img_num,
            'Total Generated Labels': total_label_num
        })

        # Step 3: Train YOLO models
        weights = train_yolo_sdal(
            paths, hyp, weights, PROJECT_DIR, log_dir, logger, cycle=i + 1,
            mode=config.get('mode', 'sequential'),
            checkpoint=checkpoint,
            save_checkpoint_func=save_checkpoint
        )

        checkpoint["current_cycle"] = i + 1
        save_checkpoint(checkpoint)
        logger.info(f"Completed cycle {i + 1}.")

        cycle_end_time = time.time()
        cycle_duration = cycle_end_time - cycle_start_time
        total_duration = cycle_end_time - start_time
        logger.info({
            'Cycle Duration (seconds)': cycle_duration,
            'Total Duration (seconds)': total_duration
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SDAL')
    parser.add_argument('--hyp', type=str, help='Path to hyp.yaml file')
    parser.add_argument('--paths', type=str, help='Path to paths.yaml file')
    parser.add_argument('--mode', type=str, choices=['sequential', 'combined'], default='sequential', help='Training mode: sequential or combined')
    parser.add_argument('--resume', action='store_true', help='Resume from the most recent checkpoint if available')
    args = parser.parse_args()

    hyp = load_yaml(args.hyp)
    paths = load_yaml(args.paths)
    experiment_name = f"{datetime.now().strftime('%y-%m-%d-%H-%M')}_SDAL_{os.path.basename(args.hyp).split('.')[0]}"
    config = {'hyp': hyp, 'paths': paths, 'exp_name': experiment_name, 'mode': args.mode, 'resume': args.resume}

    SDAL(config)
