"""
SDAL Mode-B: Day-by-Day Prequential Evaluation Experiment

This script implements the original SDAL (database-retrieval based) with a day-by-day
streaming evaluation protocol. It simulates 10 days of site deployment where:
- Each morning: evaluate on new day's data (pre-adaptation)
- Each night: mine failure cases, generate synthetic data, train combined model
- Track progress on global holdout set

Usage:
    python SDAL_modeB.py --hyp cfg/modeB_hyp.yaml --paths cfg/modeB_paths.yaml
"""

import logging
import sys
import os
import time
import yaml
import json
import shutil
from pathlib import Path
import subprocess
import warnings
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch

# Add YOLOv7 directory to Python path
sys.path.append('./yolov7')

from synthetic_oracle import oracle
from sdal_utils.sdal_utils import get_uncertain_images, merge_images, clean_stored_cache, load_yaml

warnings.filterwarnings("ignore")


def setup_logger(log_dir: Path, run_id: str) -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('SDAL_modeB')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_dir / f'sdal_modeB_{run_id}.log')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    
    return logger


def save_json(obj: Dict, path: Path):
    """Save dictionary as JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def save_yaml(obj: Dict, path: Path):
    """Save dictionary as YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f)


def yolo_eval(
    data_yaml: str,
    weights: str,
    device: str,
    img_size: int,
    batch_size: int,
    conf_thres: float,
    iou_thres: float,
    task: str,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Run YOLO evaluation and return metrics.
    """
    from yolov7.test_sdal import test
    
    logger.info(f"Running YOLO eval: task={task}, weights={weights}")
    
    mp, mr, map50, mapv, loss, maps, t, image_losses = test(
        data_yaml,
        weights=weights,
        confidence_based=False,
        batch_size=batch_size,
        imgsz=img_size,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        save_json=False,
        single_cls=False,
        augment=False,
        verbose=False,
        model=None,
        dataloader=None,
        save_txt=False,
        save_hybrid=False,
        save_conf=False,
        plots=False,
        wandb_logger=None,
        half_precision=True,
        trace=False,
        is_coco=False,
        v5_metric=False,
        training=False,
        device=device,
        task=task,
    )
    
    return {
        'mp': float(mp),
        'mr': float(mr),
        'map50': float(map50),
        'map': float(mapv),
        'loss_box': float(loss[0]) if isinstance(loss, (list, tuple)) and len(loss) > 0 else None,
        'loss_obj': float(loss[1]) if isinstance(loss, (list, tuple)) and len(loss) > 1 else None,
        'loss_cls': float(loss[2]) if isinstance(loss, (list, tuple)) and len(loss) > 2 else None,
    }


def mine_failure_cases(
    data_yaml: str,
    weights: str,
    num_cases: int,
    img_size: int,
    logger: logging.Logger,
    confidence_based: bool = False
) -> List[Tuple[str, float]]:
    """
    Mine failure cases using loss-based ranking.
    Returns list of (image_stem, loss_value) tuples.
    """
    logger.info(f"Mining {num_cases} failure cases using {'confidence' if confidence_based else 'loss'}-based ranking")
    
    mp, mr, map50, mapv, loss, maps, t, failure_cases = get_uncertain_images(
        data_yaml,
        weights,
        img_num=num_cases,
        imgsz=img_size,
        confidence_based=confidence_based
    )
    
    logger.info(f"Mining complete: mAP={mapv:.4f}, mAP50={map50:.4f}")
    logger.info(f"Selected {len(failure_cases)} failure cases")
    
    return failure_cases


def generate_synthetic_data(
    failure_cases: List[Tuple[str, float]],
    mine_images_dir: Path,
    synth_output_dir: Path,
    paths: Dict,
    hyp: Dict,
    logger: logging.Logger
) -> Tuple[int, int]:
    """
    Generate synthetic data for failure cases using the oracle.
    Returns (num_images, num_labels) generated.
    """
    total_images = 0
    total_labels = 0
    
    synth_output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_stem, loss_val in failure_cases:
        # Find the image file
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = mine_images_dir / f"{img_stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            logger.warning(f"Could not find image for stem: {img_stem}")
            continue
        
        try:
            logger.info(f"Processing failure case: {img_stem} (loss={loss_val:.4f})")
            
            labels, example_images = oracle(
                img_path,
                output_path=synth_output_dir,
                image_size=int(hyp.get('image_size', 416)),
                hdf5_path=paths.get('hdf5_path', './features/features_DenseNet201.hdf5'),
                top_k=hyp.get('top_k', 3),
                decomposer_iterations=hyp.get('decomposer_iterations', 3),
                synth_generation_premutation=hyp.get('synth_generation_permutation', 3),
                save_yolo_labels=True,
                generation_framerate=hyp.get('generation_framerate', 75),
                num_containers=hyp.get('num_containers', 3),
                blender_bin=hyp.get('blender_bin', '/snap/bin/blender'),
                oracle_runs_root=str((synth_output_dir.parent / 'oracle_runs').resolve()),
                keep_oracle_artifacts=bool(hyp.get('keep_oracle_artifacts', False)),
                failure_case_id=img_stem,
                data_gen_env_dir=paths.get('data_gen_env_dir', './sdal_utils/Data_Generator'),
                dataset_used_dir=paths.get('dataset_used_dir', './sdal_utils/Data_Generator/Dataset_used'),
                avatars_dir=paths.get('avatar_dir', './sdal_utils/Data_Generator/Avatars'),
                scenes_dir=paths.get('scene_dir', './sdal_utils/Data_Generator/Scenes'),
                scene_collection_dir=paths.get('old_scene_collection_dir', './sdal_utils/Data_Generator/Old_3DAssets/Scenes'),
                logger=logger
            )
            
            # Count generated data
            for label_dict in labels:
                total_images += len(label_dict.keys())
                total_labels += sum(len(v) for v in label_dict.values())
            
            logger.info(f"Generated data for {img_stem}")
            
        except Exception as e:
            logger.error(f"Oracle failed for {img_stem}: {str(e)}")
            continue
    
    return total_images, total_labels


def build_combined_data_yaml(
    warmup_train_path: str,
    accumulated_real_paths: List[str],
    accumulated_synth_paths: List[str],
    val_path: str,
    output_yaml: Path,
    nc: int = 1,
    names: List[str] = None
) -> Path:
    """
    Build a combined training data YAML for combined mode training.
    """
    if names is None:
        names = ['worker']
    
    # Combine all training paths
    train_paths = [warmup_train_path] + accumulated_real_paths + accumulated_synth_paths
    
    data_config = {
        'train': train_paths,
        'val': val_path,
        'test': val_path,
        'nc': nc,
        'names': names,
    }
    
    save_yaml(data_config, output_yaml)
    return output_yaml


def train_yolo(
    data_yaml: Path,
    weights: Path,
    hyp: Dict,
    project_dir: Path,
    run_name: str,
    logger: logging.Logger
) -> Path:
    """
    Train YOLOv7 model and return path to best weights.
    """
    # Clean cache files
    try:
        clean_stored_cache(str(data_yaml))
    except Exception as e:
        logger.warning(f"Could not clean cache: {e}")
    
    epochs = hyp.get('combined_epochs', 150)
    batch_size = hyp.get('batch_size', 128)
    img_size = hyp.get('image_size', 416)
    device = str(hyp.get('device', 0))
    workers = hyp.get('workers', 8)
    
    cmd = [
        'python', 'yolov7/train.py',
        '--img-size', str(img_size),
        '--batch', str(batch_size),
        '--epochs', str(epochs),
        '--data', str(data_yaml),
        '--weights', str(weights),
        '--single-cls',
        '--project', str(project_dir),
        '--name', run_name,
        '--device', device,
        '--workers', str(workers),
    ]
    
    logger.info(f"Starting YOLO training: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            env=os.environ.copy()
        )
        logger.info("Training completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e.stderr}")
        raise
    
    # Find best weights
    best_weights = project_dir / run_name / 'weights' / 'best.pt'
    if not best_weights.exists():
        logger.warning("best.pt not found, checking for last.pt")
        best_weights = project_dir / run_name / 'weights' / 'last.pt'
    
    if not best_weights.exists():
        raise FileNotFoundError(f"No weights found in {project_dir / run_name / 'weights'}")
    
    return best_weights


def run_day(
    day: int,
    current_weights: Path,
    accumulated_real: List[str],
    accumulated_synth: List[str],
    paths: Dict,
    hyp: Dict,
    run_dir: Path,
    logger: logging.Logger
) -> Tuple[Path, Dict]:
    """
    Run a single day of the Mode-B experiment.
    
    Returns:
        (new_weights_path, day_metrics)
    """
    day_dir = run_dir / f"day{day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"{'='*60}")
    logger.info(f"DAY {day:02d} STARTING")
    logger.info(f"{'='*60}")
    
    day_metrics = {
        'day': day,
        'weights_in': str(current_weights),
    }
    
    # Load day-specific data YAML
    day_data_yaml = Path(paths['data_cfg_dir']) / f"data_day{day:02d}.yaml"
    if not day_data_yaml.exists():
        raise FileNotFoundError(f"Day data YAML not found: {day_data_yaml}")
    
    day_data = load_yaml(str(day_data_yaml))
    mine_images_dir = Path(day_data['val'])
    
    # ========== MORNING: Pre-adaptation evaluation ==========
    logger.info("MORNING: Pre-adaptation evaluation")
    
    # Eval on day's field data
    field_pre = yolo_eval(
        str(day_data_yaml),
        str(current_weights),
        str(hyp.get('device', 0)),
        hyp.get('image_size', 416),
        hyp.get('eval_batch_size', 8),
        hyp.get('conf_thres', 0.001),
        hyp.get('iou_thres', 0.6),
        'test',
        logger
    )
    day_metrics['field_eval_pre'] = field_pre
    save_json({'day': day, 'split': 'field', 'phase': 'pre', 'metrics': field_pre},
              day_dir / 'field_eval_pre.json')
    logger.info(f"Field pre-eval: mAP={field_pre['map']:.4f}, mAP50={field_pre['map50']:.4f}")
    
    # Eval on global holdout
    holdout_pre = yolo_eval(
        paths['holdout_data_yaml'],
        str(current_weights),
        str(hyp.get('device', 0)),
        hyp.get('image_size', 416),
        hyp.get('eval_batch_size', 8),
        hyp.get('conf_thres', 0.001),
        hyp.get('iou_thres', 0.6),
        'test',
        logger
    )
    day_metrics['holdout_pre'] = holdout_pre
    save_json({'day': day, 'split': 'holdout', 'phase': 'pre', 'metrics': holdout_pre},
              day_dir / 'holdout_pre.json')
    logger.info(f"Holdout pre-eval: mAP={holdout_pre['map']:.4f}, mAP50={holdout_pre['map50']:.4f}")
    
    # ========== NIGHT: Mining, generation, training ==========
    logger.info("NIGHT: Mining failure cases")
    
    # Mine failure cases using loss-based approach
    failure_cases = mine_failure_cases(
        str(day_data_yaml),
        str(current_weights),
        hyp.get('failure_cases_per_day', 20),
        hyp.get('image_size', 416),
        logger,
        confidence_based=hyp.get('confidence_based', False)
    )
    
    day_metrics['failure_cases'] = [{'stem': s, 'loss': float(l)} for s, l in failure_cases]
    save_json({'day': day, 'failure_cases': day_metrics['failure_cases']},
              day_dir / 'mining_log.json')
    
    # Generate synthetic data
    logger.info("NIGHT: Generating synthetic data")
    synth_day_dir = day_dir / 'synth_train'
    
    num_images, num_labels = generate_synthetic_data(
        failure_cases,
        mine_images_dir,
        synth_day_dir,
        paths,
        hyp,
        logger
    )
    
    day_metrics['synth_generated'] = {'images': num_images, 'labels': num_labels}
    logger.info(f"Generated {num_images} synthetic images with {num_labels} labels")
    
    # Update accumulated paths
    # Add mining pool to accumulated real data
    accumulated_real.append(str(mine_images_dir))
    
    # Add synthetic data if generated
    synth_images_dir = synth_day_dir / 'images'
    if synth_images_dir.exists() and any(synth_images_dir.iterdir()):
        accumulated_synth.append(str(synth_images_dir))
    
    # Build combined training YAML
    logger.info("NIGHT: Building combined training data")
    combined_yaml = day_dir / 'data_combined.yaml'
    
    # Get validation path from holdout data
    holdout_data = load_yaml(paths['holdout_data_yaml'])
    val_path = holdout_data.get('val', holdout_data.get('train'))
    
    build_combined_data_yaml(
        paths['warmup_train_images'],
        accumulated_real.copy(),
        accumulated_synth.copy(),
        val_path,
        combined_yaml
    )
    
    day_metrics['train_data'] = {
        'warmup': paths['warmup_train_images'],
        'accumulated_real': accumulated_real.copy(),
        'accumulated_synth': accumulated_synth.copy(),
    }
    
    # Train combined model
    logger.info("NIGHT: Training combined model")
    yolo_project = day_dir / 'yolov7_train'
    run_name = f"day{day:02d}_combined"
    
    new_weights = train_yolo(
        combined_yaml,
        current_weights,
        hyp,
        yolo_project,
        run_name,
        logger
    )
    
    # Copy best weights to day directory
    weights_dir = day_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    final_weights = weights_dir / 'best.pt'
    shutil.copy2(new_weights, final_weights)
    
    day_metrics['weights_out'] = str(final_weights)
    
    # ========== NIGHT: Post-adaptation holdout evaluation ==========
    logger.info("NIGHT: Post-adaptation evaluation")
    
    holdout_post = yolo_eval(
        paths['holdout_data_yaml'],
        str(final_weights),
        str(hyp.get('device', 0)),
        hyp.get('image_size', 416),
        hyp.get('eval_batch_size', 8),
        hyp.get('conf_thres', 0.001),
        hyp.get('iou_thres', 0.6),
        'test',
        logger
    )
    day_metrics['holdout_post'] = holdout_post
    save_json({'day': day, 'split': 'holdout', 'phase': 'post', 'metrics': holdout_post},
              day_dir / 'holdout_post.json')
    logger.info(f"Holdout post-eval: mAP={holdout_post['map']:.4f}, mAP50={holdout_post['map50']:.4f}")
    
    # Calculate improvement
    improvement = holdout_post['map'] - holdout_pre['map']
    day_metrics['holdout_improvement'] = improvement
    logger.info(f"Day {day:02d} improvement: {improvement:+.4f} mAP")
    
    # Save day summary
    save_json(day_metrics, day_dir / 'day_summary.json')
    
    return final_weights, day_metrics


def run_experiment(hyp: Dict, paths: Dict, run_id: str):
    """
    Run the complete Mode-B experiment.
    """
    repo_root = Path(__file__).resolve().parent

    # Setup directories
    experiments_root = Path(paths.get('experiments_root', './experiments/modeB_runs'))
    if not experiments_root.is_absolute():
        experiments_root = (repo_root / experiments_root).resolve()
    run_dir = experiments_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(paths.get('log_dir', './logs/modeB'))
    if not log_dir.is_absolute():
        log_dir = (repo_root / log_dir).resolve()
    logger = setup_logger(log_dir, run_id)
    
    logger.info(f"Starting SDAL Mode-B Experiment")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run directory: {run_dir}")
    
    # Save config
    save_yaml(hyp, run_dir / 'hyp.yaml')
    save_yaml(paths, run_dir / 'paths.yaml')
    
    # Initialize
    num_days = hyp.get('num_days', 10)
    current_weights = Path(paths['warmed_up_model_weights'])
    
    if not current_weights.exists():
        raise FileNotFoundError(f"Warmup weights not found: {current_weights}")
    
    logger.info(f"Initial weights: {current_weights}")
    logger.info(f"Number of days: {num_days}")
    
    # Track accumulated data paths
    accumulated_real: List[str] = []
    accumulated_synth: List[str] = []
    
    # Track all metrics
    all_metrics = {
        'run_id': run_id,
        'start_time': datetime.now().isoformat(),
        'config': {'hyp': hyp, 'paths': paths},
        'days': [],
    }
    
    start_time = time.time()
    
    # Run day-by-day loop
    for day in range(1, num_days + 1):
        day_start = time.time()
        
        try:
            current_weights, day_metrics = run_day(
                day,
                current_weights,
                accumulated_real,
                accumulated_synth,
                paths,
                hyp,
                run_dir,
                logger
            )
            
            day_metrics['duration_seconds'] = time.time() - day_start
            all_metrics['days'].append(day_metrics)
            
            # Save running results
            save_json(all_metrics, run_dir / 'results.json')
            
        except Exception as e:
            logger.error(f"Day {day} failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Save partial results
            all_metrics['error'] = {'day': day, 'message': str(e)}
            save_json(all_metrics, run_dir / 'results.json')
            raise
    
    # Final summary
    total_time = time.time() - start_time
    all_metrics['end_time'] = datetime.now().isoformat()
    all_metrics['total_duration_seconds'] = total_time
    
    # Calculate overall improvement
    if all_metrics['days']:
        first_day = all_metrics['days'][0]
        last_day = all_metrics['days'][-1]
        overall_improvement = last_day['holdout_post']['map'] - first_day['holdout_pre']['map']
        all_metrics['overall_improvement'] = overall_improvement
        logger.info(f"Overall improvement: {overall_improvement:+.4f} mAP")
    
    save_json(all_metrics, run_dir / 'results.json')
    
    logger.info(f"Experiment complete!")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Results saved to: {run_dir / 'results.json'}")


def main():
    parser = argparse.ArgumentParser(description='SDAL Mode-B Day-by-Day Experiment')
    parser.add_argument('--hyp', type=str, required=True, help='Path to hyperparameters YAML')
    parser.add_argument('--paths', type=str, required=True, help='Path to paths YAML')
    parser.add_argument('--run-id', type=str, default=None, help='Run ID (auto-generated if not provided)')
    args = parser.parse_args()
    
    # Load configs
    hyp = load_yaml(args.hyp)
    paths = load_yaml(args.paths)
    
    # Generate run ID
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run experiment
    run_experiment(hyp, paths, run_id)


if __name__ == '__main__':
    main()
