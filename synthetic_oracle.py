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
import json
from sdal_utils.sdal_utils import create_data_gen_env
from sdal_utils.pickle2yolo import pickle2yolo
from sdal_utils.blender_parallel_runner import run_parallel_generators
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
        num_containers=3,
        run_root: Path = None,
        blender_bin: str = "blender",
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
        num_containers=num_containers,
        run_root=run_root,
        blender_bin=blender_bin,
        logger=logger)
    
    return True

def generate_yolo_labels(input_folders, output_path, save_txt, logger=None):
    labels = []
    example_images = {}
    for input_folder in input_folders:
        try:
            if not os.path.exists(os.path.join(output_path, 'labels')):
                os.makedirs(os.path.join(output_path, 'labels'))
            label = pickle2yolo(input_folder, os.path.join(output_path, 'labels'), save_txt=save_txt)
            if logger:
                logger.info(f"Generated {len(label.values())} number of YOLO labels for {input_folder}")
            labels.append(label)
        except Exception as e:
            if logger:
                logger.error(f"Error generating YOLO labels for {input_folder}: {str(e)}")
            continue
        
        if logger:
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
        num_containers=3,
        blender_bin: str = "blender",
        oracle_runs_root: str = None,
        keep_oracle_artifacts: bool = False,
        failure_case_id: str = None,
        hdf5_path='./features/features_DenseNet201.hdf5', 
        data_gen_env_dir='./sdal_utils/Data_Generator', 
        save_yolo_labels=True, 
        dataset_used_dir='./sdal_utils/Data_Generator/Dataset_used', 
        logger=None,
        ) -> tuple:
    
    data_gen_env_dir = Path(data_gen_env_dir).resolve()
    output_path = Path(output_path)
    dataset_used_dir = Path(dataset_used_dir)

    def _unique_run_dir(root: Path, base: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in base).strip("_") or "failure_case"
        candidate = root / safe
        if not candidate.exists():
            return candidate
        i = 1
        while (root / f"{safe}-{i}").exists():
            i += 1
        return root / f"{safe}-{i}"

    run_root = None
    input_folders = []
    try:
        # Decide run-scoped artifact root
        if oracle_runs_root is None:
            oracle_runs_root = str(output_path / "_oracle_runs")
        oracle_runs_root = Path(oracle_runs_root).resolve()

        base_id = failure_case_id or Path(query_image_path).stem
        run_root = _unique_run_dir(oracle_runs_root, base_id).resolve()
        run_root.mkdir(parents=True, exist_ok=True)

        if logger:
            logger.info(f"[oracle] run_root={run_root}")

        # Stage 1: decomposer writes config.json + normalization into run_root
        decomposed = decomposer(
            query_image=query_image_path,
            data_gen_env_dir=data_gen_env_dir,
            hdf5_path=hdf5_path,
            generation_framerate=generation_framerate,
            top_k=top_k,
            image_size=image_size,
            decomposer_iterations=decomposer_iterations,
            synth_generation_premutation=synth_generation_premutation,
            num_containers=num_containers,
            run_root=run_root,
            blender_bin=blender_bin,
            avatars_dir=avatars_dir,
            scenes_dir=scenes_dir,
            scene_collection_dir=scene_collection_dir,
            logger=logger,
        )
        if not decomposed:
            raise ValueError("Decomposition failed")

        config_json = run_root / "config.json"
        if not config_json.exists():
            raise FileNotFoundError(f"Expected config.json not found at {config_json}")

        # Stage 2: run Blender generators in parallel (no Docker)
        manifest = run_parallel_generators(
            blender_bin=blender_bin,
            empty_blend=(data_gen_env_dir / "Empty.blend").resolve(),
            generator_py=(data_gen_env_dir / "231109_Adaptive_Data_Generator.py").resolve(),
            config_json=config_json.resolve(),
            run_root=run_root,
            data_gen_env_dir=data_gen_env_dir.resolve(),
            num_workers=int(num_containers),
            logger=logger,
        )

        # Collect generated sequence folders
        for wid in range(1, int(num_containers) + 1):
            dataset_dir = run_root / f"worker_{wid:02d}" / "Dataset"
            if not dataset_dir.exists():
                continue
            for p in dataset_dir.iterdir():
                if p.is_dir():
                    input_folders.append(p)

        if not input_folders:
            raise ValueError(
                f"Synthetic generation produced no sequences. Inspect {run_root}/manifest.json and worker logs."
            )

        if logger:
            logger.info(f"[oracle] generating YOLO labels from {len(input_folders)} sequences into {output_path}")

        labels, example_images = generate_yolo_labels(input_folders, output_path, save_yolo_labels, logger=logger)

        # Cleanup raw datasets if requested
        if not keep_oracle_artifacts:
            for wid in range(1, int(num_containers) + 1):
                dataset_dir = run_root / f"worker_{wid:02d}" / "Dataset"
                if dataset_dir.exists():
                    shutil.rmtree(dataset_dir, ignore_errors=True)
            if logger:
                logger.info(f"[oracle] cleaned raw datasets under {run_root} (kept logs + manifest)")
        else:
            if logger:
                logger.info(f"[oracle] keeping full raw artifacts under {run_root}")

        # dataset_used_dir is kept for backward compatibility but no longer used in docker-free mode
        if logger and dataset_used_dir:
            logger.debug(f"[oracle] dataset_used_dir={dataset_used_dir} (not used in docker-free mode)")

        return labels, example_images

    except Exception as e:
        if logger:
            logger.error(f"Error in oracle function: {str(e)}")
            if run_root is not None:
                logger.error(f"[oracle] artifacts retained at: {run_root}")
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
