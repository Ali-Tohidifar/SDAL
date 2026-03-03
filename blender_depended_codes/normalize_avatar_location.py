import json
import os  # Import os module for file deletion
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

def _read_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    config_file = Path(config_file)
    if config_file.suffix.lower() == ".json":
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _write_config(data: Dict[str, Any], config_file: Union[str, Path]) -> None:
    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    if config_file.suffix.lower() == ".json":
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

def write_json(data, json_file):
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

def read_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def process_points_with_blender(scene_dir, input_json, output_json, blender_bin: str = "blender"):
    # command = [
    #     "blender", "--background", "--python", "get_pose.py",
    #     "--", "--input_json", input_json, "--output_file", output_json
    # ]

    repo_root = Path(__file__).resolve().parents[1]
    get_pose_py = repo_root / "blender_depended_codes" / "get_pose.py"
    command = [
        str(blender_bin),
        "--background",
        "--python",
        str(get_pose_py),
        "--",
        "--scene_dir",
        str(scene_dir),
        "--input_json",
        str(input_json),
        "--output_file",
        str(output_json),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # import ipdb;ipdb.set_trace()

def update_config_with_results(config_file, results):
    data = _read_config(config_file)

    for result in results:
        label, bone_point = result["point"], result["normalized_position"]
        
        # process labels
        if label == "camera":
            data['camera_location'] = bone_point
        else:
            # Split label back into worker and bone for updating
            worker, bone = label.split('|')
            data['worker_bones_info'][worker][bone] = bone_point

    _write_config(data, config_file)

def get_scene_dir(config_data, scene_collection_dir = r'E:\Data_Generator_WorkerDetection\Old_3DAssets\Scenes'):
    scene_name = config_data['scene_name']
    scene_dir = None
    for file in Path(scene_collection_dir).iterdir():
        if scene_name in file.name:
            scene_dir = file
            break
    if scene_dir is None:
        new_path = Path(scene_collection_dir) / 'New folder'
        for file in new_path.iterdir():
            if scene_name in file.name:
                scene_dir = file
                break
    
    if scene_dir is None:
        raise FileNotFoundError(f"Scene not found in {scene_collection_dir}")
    return scene_dir

def normalize_all_location(
    scene_collection_dir,
    config_file: Union[str, Path] = './Data_Generator/config.yaml',
    input_json: Union[str, Path] = 'input_points.json',
    output_json: Union[str, Path] = 'output_vectors.json',
    blender_bin: str = "blender",
    ):
    config_file = Path(config_file)
    cfg = _read_config(config_file)

    points = []
    # add workers locations
    for worker, bones in cfg['worker_bones_info'].items():
        for bone, position in bones.items():
            points.append({
                "label": f"{worker}|{bone}",
                "position": position
            })

    # import ipdb;ipdb.set_trace()
    input_json = Path(input_json)
    output_json = Path(output_json)

    write_json(points, input_json)
    scene_dir = get_scene_dir(cfg, scene_collection_dir=scene_collection_dir)
    process_points_with_blender(scene_dir, input_json, output_json, blender_bin=blender_bin)
    results = read_json(output_json)

    # Delete temp json files after reading
    try:
        os.remove(output_json)
    except FileNotFoundError:
        pass
    try:
        os.remove(input_json)
    except FileNotFoundError:
        pass

    update_config_with_results(config_file, results)
    return True

