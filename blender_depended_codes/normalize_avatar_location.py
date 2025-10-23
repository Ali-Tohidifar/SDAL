import yaml
import json
import subprocess
from pathlib import Path
import os  # Import os module for file deletion

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data

def write_yaml(data, yaml_file):
    with open(yaml_file, 'w') as file:
        yaml.safe_dump(data, file)

def write_json(data, json_file):
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

def read_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def process_points_with_blender(scene_dir, input_json, output_json):
    # command = [
    #     "blender", "--background", "--python", "get_pose.py",
    #     "--", "--input_json", input_json, "--output_file", output_json
    # ]

    command = f'blender --background --python ./blender_depended_codes/get_pose.py -- --scene_dir "{scene_dir}" --input_json "{input_json}" --output_file "{output_json}"'
    # result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    subprocess.run(command, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # import ipdb;ipdb.set_trace()

def update_yaml_with_results(yaml_file, results):
    data = read_yaml(yaml_file)

    for result in results:
        label, bone_point = result["point"], result["normalized_position"]
        
        # process labels
        if label == "camera":
            data['camera_location'] = bone_point
        else:
            # Split label back into worker and bone for updating
            worker, bone = label.split('|')
            data['worker_bones_info'][worker][bone] = bone_point

    write_yaml(data, yaml_file)

def get_scene_dir(yaml_data, scene_collection_dir = r'E:\Data_Generator_WorkerDetection\Old_3DAssets\Scenes'):
    scene_name = yaml_data['scene_name']
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
    yaml_file = './Data_Generator/config.yaml',
    input_json = 'input_points.json',
    output_json = 'output_vectors.json',
    ):
    yaml_data = read_yaml(yaml_file)

    points = []
    # add workers locations
    for worker, bones in yaml_data['worker_bones_info'].items():
        for bone, position in bones.items():
            points.append({
                "label": f"{worker}|{bone}",
                "position": position
            })

    # import ipdb;ipdb.set_trace()
    write_json(points, input_json)
    scene_dir = get_scene_dir(yaml_data, scene_collection_dir=scene_collection_dir)
    process_points_with_blender(scene_dir, input_json, output_json)
    results = read_json(output_json)
    os.remove(output_json)  # Delete the output JSON file after reading
    os.remove(input_json)  # Delete the input JSON file after processing
    update_yaml_with_results(yaml_file, results)
    return True

