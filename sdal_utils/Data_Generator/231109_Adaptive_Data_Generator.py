import argparse
import json
import logging
import bpy
import bmesh
import math
import random
from mathutils import Vector
import pickle
import os
import bpy_extras
from pathlib import Path
import mathutils
from datetime import datetime, timezone
import zoneinfo

# Your current timestamp in UTC
timestamp_utc = datetime.now(timezone.utc)
# Define the EDT timezone
edt_timezone = zoneinfo.ZoneInfo('America/New_York')
# Convert the timestamp to EDT
timestamp_edt = timestamp_utc.astimezone(edt_timezone)
timestamp = timestamp_edt.strftime('%y-%m-%d-%H-%M')

def _parse_blender_args():
    """
    Blender passes its own args; our args start after '--'.
    This keeps backward compatibility: if args are not provided we fall back to
    reading config from ./config.yaml and writing into ./Dataset and ./logs.
    """
    import sys
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--logs-dir", type=str, default=None)
    parser.add_argument("--worker-id", type=int, default=None)
    return parser.parse_args(argv)


_args = _parse_blender_args()
WORKER_ID = int(_args.worker_id) if _args.worker_id is not None else None

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AdaptiveDataGenerator')
logger.setLevel(logging.DEBUG)

# Ensure log messages are saved to a file (run-scoped logs dir if provided)
logs_dir = Path(_args.logs_dir).resolve() if _args.logs_dir else Path(os.getcwd()).resolve() / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_suffix = f"_w{WORKER_ID:02d}" if WORKER_ID is not None else ""
log_file = logs_dir / f"data_generation_{timestamp}{log_suffix}.log"
fh = logging.FileHandler(str(log_file))
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info('Starting data generation process.')


Code_dir = Path(os.getcwd())
# Code_dir = Path(os.path.join(os.getcwd(),'./sdal_utils/Data_Generator'))

"Read config and user inputs"
# Read config json file (preferred) or fall back to legacy YAML if present
config = None
if _args.config_json:
    cfg_path = Path(_args.config_json).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"Loaded config JSON: {cfg_path}")
else:
    # Legacy fallback: config.yaml in current working directory
    try:
        import yaml  # lazy import (only for legacy mode)
        with open(Code_dir / "config.yaml", "r") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        logger.warning("Using legacy config.yaml (no --config-json provided).")
    except Exception as exc:
        raise RuntimeError("No --config-json provided and legacy config.yaml could not be loaded.") from exc

# Setup directories
Dataset_dir = Path(_args.dataset_dir).resolve() if _args.dataset_dir else (Code_dir / 'Dataset')
Dataset_dir.mkdir(parents=True, exist_ok=True)
Avatar_dir = Code_dir / 'Avatars'
Scene_dir= Code_dir / 'Scenes'
report_path = Code_dir / 'Report.txt'
horizon_path = Code_dir / 'Horizon.blend'
empty_dir = Code_dir / 'Empty.blend'

# workers in scene
Workers_in_Scene = [worker + '.blend' for worker in config['worker_bones_info']]
avatar_paths = [Avatar_dir / avatar for avatar in Workers_in_Scene]
avatar_paths = [path for path in avatar_paths if os.path.isfile(path)]

avatars_locations = config['worker_bones_info']
Number_of_Workers = len(config['worker_bones_info'])

# target_avatar
target_rig_name = config['target_avatar']

# Scene_path
scene_name = config['scene_name'] + '.blend'
scene_path = Scene_dir / scene_name

# Render Setting
Number_of_Image_Sequences =  int(config['Number_of_Image_Sequences'])
sun_elevation = config['lighting']['sun_elevation']
sun_state = config['lighting']['sun_state']
threshold = float(config['Threshold'])
framerate = int(config['Framerate'])

# Initialize the scene
scene = bpy.context.scene
camera = bpy.context.scene.camera

# Camear location record
camera_record = {'distance': config['distance'], 'orientation': config['orientation']}


"""
Setup GPU in Blender
"""
logger.info('Setting up GPU in Blender.')
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
devices = prefs.get_devices_for_type('CUDA')
for device in devices:
    if 'RTX' in device['name']:
        device.use = True  # enable any available GPU
        logger.info(f'Using {device["name"]} for rendering synthetic images')
    else:
        device.use = False
for device in devices:
    logger.info(f'{device["name"]}, use: {device["use"]}')
    
    

"""
Functions
"""

"Location Noramlization"
def normalized2actual(normalized_pos, obj_reference):
    """
    Converts a normalized position to actual position based on the bounding box of the object.

    Args:
        normalized_pos (Vector): The normalized position to convert.
        obj_reference (Object): The reference object to calculate the bounding box.

    Returns:
        Vector: The actual position in world coordinates.
    """
    
    # Update the scene to get current data
    bpy.context.view_layer.update()

    # Calculate the new bounding box dimensions and location
    bbox_corners = [obj_reference.matrix_world @ Vector(corner) for corner in obj_reference.bound_box]
    min_corner_new = Vector((min(c.x for c in bbox_corners), min(c.y for c in bbox_corners), min(c.z for c in bbox_corners)))
    max_corner_new = Vector((max(c.x for c in bbox_corners), max(c.y for c in bbox_corners), max(c.z for c in bbox_corners)))
    bbox_dimensions_new = max_corner_new - min_corner_new

    # Calculate the absolute position based on the normalized position and new bounding box
    actual_position = Vector((
        min_corner_new.x + normalized_pos.x * bbox_dimensions_new.x,
        min_corner_new.y + normalized_pos.y * bbox_dimensions_new.y,
        min_corner_new.z + normalized_pos.z * bbox_dimensions_new.z
    ))

    return actual_position

"Snapping to the workzone"
def get_nearest_workzone(obj, point):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    closest_edge = None
    min_dist = float('inf')

    for edge in bm.edges:
        p1 = obj.matrix_world @ edge.verts[0].co
        p2 = obj.matrix_world @ edge.verts[1].co
        midpoint = (p1 + p2) / 2
        dist = (midpoint - point).length
        if dist < min_dist:
            min_dist = dist
            closest_edge = edge

    if closest_edge:
        p1 = obj.matrix_world @ closest_edge.verts[0].co
        p2 = obj.matrix_world @ closest_edge.verts[1].co
        return (p1 + p2) / 2
    return None


"Camera Localization Function"
def localize_camera_based_on_record(record, camera_name='Camera', avatar_name='Armature'):
    # Get the camera and avatar objects
    camera = bpy.data.objects[camera_name]
    avatar = bpy.data.objects[avatar_name]

    # Retrieve the distance and orientation from the record
    distance = record['distance']
    orientation = mathutils.Euler(record['orientation'])

    # Calculate the direction vector from the orientation
    direction = mathutils.Vector((0, 0, -1))
    direction.rotate(orientation)

    # Set the new location of the camera
    camera.location = avatar.location + direction * distance

    # Set the orientation of the camera
    camera.rotation_euler = orientation


"Camera location perturbation function"
def perturb_orientation(record, target_rig_name, target_rig, max_attempts=200, perturbation_magnitude=0.4):
    attempt = 0
    original_orientation = mathutils.Euler(record['orientation'])
    while attempt < max_attempts:
        # Perturb the orientation angles slightly
        perturbed_orientation = mathutils.Euler((
            original_orientation.x + random.uniform(-perturbation_magnitude, perturbation_magnitude),
            original_orientation.y + random.uniform(-perturbation_magnitude, perturbation_magnitude),
            original_orientation.z + random.uniform(-perturbation_magnitude, perturbation_magnitude)
        ))

        # Create a new record with the perturbed orientation
        new_record = {
            'distance': record['distance'],
            'orientation': (perturbed_orientation.x, perturbed_orientation.y, perturbed_orientation.z)
        }

        # Localize the camera based on the new record
        localize_camera_based_on_record(new_record, avatar_name=target_rig_name)

        # Validate the new location
        if occlusion_detector(target_rig, camera) < 1:
            logger.info(f'Camera orientation perturbed successfully. Attempt: {attempt + 1}')
            return new_record

        attempt += 1
    # import ipdb; ipdb.set_trace()
    logger.error(f'Failed to perturb camera orientation after {max_attempts} attempts. Occlusion for target rig of {target_rig_name} is {occlusion_detector(target_rig, camera)}')
    raise ValueError("Could not find a valid orientation after {} attempts".format(max_attempts))


"New Camera Function"

def new_camera(focal_len=20):
    # Removing the camera
    bpy.ops.object.select_all(action='DESELECT')
    try:
        bpy.data.objects['Camera'].select_set(True)
        if bpy.data.objects['Camera'].select_get():
            bpy.ops.object.delete()
    except:
        pass

    # create the first camera data
    bpy.ops.object.camera_add(location=(0, 0, 0),
                              rotation=(0, 0, 0))
    scene.camera = bpy.context.object
    scene.camera.data.lens = focal_len
    camera = bpy.context.scene.camera


"Oclussion detector for workers with bones"

def occlusion_detector(target_arm, origin):
    # collect all other mesh objects that can occlude target
    others = [obj for obj in bpy.data.objects if (
        obj.parent != target_arm and obj != origin and obj.type == 'MESH' and 'cam_circle' not in obj.name)]

    # add cubes in bone locations
    added_cubes = []
    for bone in target_arm.pose.bones:
        bonePos = target_arm.matrix_world @ bone.head
        bpy.ops.mesh.primitive_cube_add(
            size=0.05, enter_editmode=False, align='WORLD', location=bonePos, scale=(1, 1, 1))
        added_cubes.append(bpy.context.active_object)

    # depsgraph = bpy.context.evaluated_depsgraph_get()

    # iterate through target cubes and identify occlusion
    occlusion = 0
    for target in added_cubes:
        # calculate target hit distance
        target_in_target_space = target.matrix_world.inverted() @ target.location
        origin_in_target_space = target.matrix_world.inverted() @ origin.location
        ray_direction = target_in_target_space - origin_in_target_space

        ray_cast_target = target.ray_cast(
            origin_in_target_space, ray_direction)
        hit_loc_in_glob = target.matrix_world @ ray_cast_target[1]
        target_distance = hit_loc_in_glob - origin.location

        # loop through others and detect occlusion
        for obj in others:
            # calculate non-target hit distance
            origin_in_obj_space = obj.matrix_world.inverted() @ origin.location
            target_in_obj_space = obj.matrix_world.inverted() @ target.location
            ray_direction = target_in_obj_space - origin_in_obj_space

            ray_cast_obj = obj.ray_cast(origin_in_obj_space, ray_direction)
            hit_loc_in_glob = obj.matrix_world @ ray_cast_obj[1]
            obj_distance = hit_loc_in_glob - origin.location

            if ray_cast_obj[0] and target_distance.length > obj_distance.length:
                occlusion += 1

    # calculate occlusion percentage
    precentage = occlusion/len(added_cubes)

    # remove added cubes
    for cube in added_cubes:
        cube.select_set(True)
    bpy.ops.object.delete()
    logger.debug(f'Occlusion percentage for {target_arm.name} is {round(precentage*100,2)} percent')
    return precentage


"Joint Tracker Function"

def joint_tracker(lighting, workers_name_list, path=Dataset_dir):
    #TODO
    workers = [worker for worker in bpy.data.objects if 'Armature' in worker.name]
    ###

    # Create a dictionary for data capture
    Information_dict = {}

    # Capturing the bone names and connections of workers once and use it in each frame iteration
    bone_connection_dict = {}
    bones_list_dict = {}
    leaf_bones_dict = {}
    root_bones_dict = {}

    for worker in workers:
        # find root and leaf bones
        bones_list = []
        leaf_bones = []
        root_bones = []

        for bone in worker.pose.bones:
            bones_list.append(bone.name)
            if len(bone.children) == 0:
                leaf_bones.append(bone)
            elif bone.parent is None:
                root_bones.append(bone)
        bones_list_dict[str(worker.name)] = bones_list
        leaf_bones_dict[str(worker.name)] = [bone.name for bone in leaf_bones]
        root_bones_dict[str(worker.name)] = [bone.name for bone in root_bones]

        # start from leaf bone and iterate through parents and append connections
        sklet_set = set()    # using set here to aviod duplications
        for bone in leaf_bones:
            iter_bone = bone
            while iter_bone.parent != None:
                sklet_set.add((iter_bone.name, iter_bone.parent.name))
                iter_bone = iter_bone.parent
        bone_connection_dict[str(worker.name)] = list(sklet_set)

    logger.info('Ground truths are being extracted:')
    total = bpy.context.scene.frame_end+1 - bpy.context.scene.frame_start
    # Looping over the animation frames
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end+1, bpy.context.scene.frame_step):
        # print progress precentage
        progress = int(frame/total*100)
        progress_report = '%' + str(progress)
        # logger.info(f'extracted {progress_report} percent')
        
        # setting scene to the required frame
        bpy.context.scene.frame_set(frame)

        # define an intermidate dictionaries for each of the frames
        info_on_each_frame = {}

        # calculate render size
        render_scale = bpy.context.scene.render.resolution_percentage / 100
        render_size = (int(bpy.context.scene.render.resolution_x * render_scale),
                       int(bpy.context.scene.render.resolution_y * render_scale))

        # save render size in dictionary
        info_on_each_frame['render_size'] = render_size

        # save camera location
        info_on_each_frame['camera_location'] = list(
            bpy.context.scene.camera.location)

        # loop over workers
        for worker in workers:
            # creating worker name tag
            rig_name = worker.name.replace('Armature: ', '')

            # define an intermidate dictionaries for each worker in each frames
            info_on_each_frame_for_each_worker = {}

            # calculate occlusion
            occlusion_percentage = occlusion_detector(
                worker, bpy.context.scene.camera)
            info_on_each_frame_for_each_worker['occlusion'] = occlusion_percentage

            # add bone names to info dict
            info_on_each_frame_for_each_worker['bone_name'] = bones_list_dict[str(
                worker.name)]

            # add root_bones to info dict
            info_on_each_frame_for_each_worker['root_bones'] = root_bones_dict[str(
                worker.name)]

            # add leaf_bones to info dict
            info_on_each_frame_for_each_worker['leaf_bones'] = leaf_bones_dict[str(
                worker.name)]

            # add bone_connection to info dict
            info_on_each_frame_for_each_worker['bone_connection'] = bone_connection_dict[str(
                worker.name)]

            # setting the intermidate dictionaries for each of the frames
            worker_bone_3d_location_dict = {}
            worker_bone_2d_location_dict = {}
            worker_bone_pixel_location_dict = {}

            # initializing the bounding boxes max and min coordinations
            bounding_box_max_x = -1*float('inf')
            bounding_box_max_y = -1*float('inf')
            bounding_box_min_x = 1*float('inf')
            bounding_box_min_y = 1*float('inf')

            # initializing the 3D bounding boxes max and min coordinations
            bounding_box_3D_max_x = -1*float('inf')
            bounding_box_3D_max_y = -1*float('inf')
            bounding_box_3D_max_z = -1*float('inf')
            bounding_box_3D_min_x = 1*float('inf')
            bounding_box_3D_min_y = 1*float('inf')
            bounding_box_3D_min_z = 1*float('inf')

            # looping over each bone
            for bone in worker.data.bones:
                # Selecting the bones. This can be head, tail, center...
                # More info: https://docs.blender.org/api/current/bpy.types.PoseBone.html?highlight=bpy%20bone#bpy.types.PoseBone.bone
                bone_pos = worker.pose.bones[bone.name].head

                # Converting bone local position to the scene coordinate
                # Since Blender 2.8 you multiply matrices with @ not with *
                bone_pos_glob = worker.matrix_world @ bone_pos

                #  create bone_location_3d dict
                worker_bone_3d_location_dict[str(bone.name)] = [
                    bone_pos_glob[0], bone_pos_glob[1], bone_pos_glob[2]]

                # Converting 3d location of iterating bone to pixel coordinates
                coordinate_2d = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, bone_pos_glob)

                #  create bone_location_3d dict
                worker_bone_2d_location_dict[str(bone.name)] = [
                    coordinate_2d[0], coordinate_2d[1], coordinate_2d[2]]

                # Filtering the pixel coordinates behind the camera
                if coordinate_2d.z < 0:
                    bounding_box_min_x = 1*float('inf')
                    bounding_box_max_x = -1*float('inf')
                    bounding_box_min_y = 1*float('inf')
                    bounding_box_max_y = -1*float('inf')

                    bounding_box_3D_max_x = -1*float('inf')
                    bounding_box_3D_max_y = -1*float('inf')
                    bounding_box_3D_max_z = -1*float('inf')
                    bounding_box_3D_min_x = 1*float('inf')
                    bounding_box_3D_min_y = 1*float('inf')
                    bounding_box_3D_min_z = 1*float('inf')

                else:
                    pixel_coordinate_x = coordinate_2d.x * render_size[0]
                    pixel_coordinate_y = render_size[1] - \
                        (coordinate_2d.y * render_size[1])

                    # Saving bone locations of each workers
                    worker_bone_pixel_location_dict[str(bone.name)] = [
                        pixel_coordinate_x, pixel_coordinate_y, coordinate_2d.z]

                    ### Fix this part ###
                    # Selecting the max and min bounding box coordinates
                    if pixel_coordinate_x < bounding_box_min_x:
                        bounding_box_min_x = pixel_coordinate_x
                    if pixel_coordinate_x > bounding_box_max_x:
                        bounding_box_max_x = pixel_coordinate_x
                    if pixel_coordinate_y < bounding_box_min_y:
                        bounding_box_min_y = pixel_coordinate_y
                    if pixel_coordinate_y > bounding_box_max_y:
                        bounding_box_max_y = pixel_coordinate_y

                    # Selecting the max and min 3D bounding box coordinates
                    if bone_pos[0] < bounding_box_3D_min_x:
                        bounding_box_3D_min_x = bone_pos[0]
                    if bone_pos[0] > bounding_box_3D_max_x:
                        bounding_box_3D_max_x = bone_pos[0]
                    if bone_pos[1] < bounding_box_3D_min_y:
                        bounding_box_3D_min_y = bone_pos[1]
                    if bone_pos[1] > bounding_box_3D_max_y:
                        bounding_box_3D_max_y = bone_pos[1]
                    if bone_pos[2] < bounding_box_3D_min_z:
                        bounding_box_3D_min_z = bone_pos[2]
                    if bone_pos[2] > bounding_box_3D_max_z:
                        bounding_box_3D_max_z = bone_pos[2]
                    ###

                # iterate through meshes and extract 3d optimas and pixel optimas for each workers

                # clamp the bounding boxes to the render size

            # writing the data to the dictionaries
            info_on_each_frame_for_each_worker['bone_location_2d_raw'] = worker_bone_2d_location_dict

            if bounding_box_min_x == 1*float('inf') or bounding_box_max_x == -1*float('inf') or bounding_box_min_y == 1*float('inf') or bounding_box_max_y == -1*float('inf') or bounding_box_3D_min_x == 1*float('inf') or bounding_box_3D_max_x == -1*float('inf') or bounding_box_3D_min_y == 1*float('inf') or bounding_box_3D_max_y == -1*float('inf') or bounding_box_3D_min_z == 1*float('inf') or bounding_box_3D_max_z == -1*float('inf'):
                continue

            else:
                # add bone info and bb info to info dict
                info_on_each_frame_for_each_worker['BB2D'] = [
                    [bounding_box_min_x, bounding_box_min_y], [bounding_box_max_x, bounding_box_max_y]]
                info_on_each_frame_for_each_worker['BB3D_global_coordinate'] = [[bounding_box_3D_min_x, bounding_box_3D_min_y, bounding_box_3D_min_z], [
                    bounding_box_3D_max_x, bounding_box_3D_max_y, bounding_box_3D_max_z]]
                info_on_each_frame_for_each_worker['bone_location_3d'] = worker_bone_3d_location_dict
                info_on_each_frame_for_each_worker['bone_location_2d'] = worker_bone_pixel_location_dict

                # Convert 3D BB in global coordinate to pixel coordinate
                p1_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_min_x, bounding_box_3D_min_y, bounding_box_3D_min_z))
                p2_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_max_x, bounding_box_3D_min_y, bounding_box_3D_min_z))
                p3_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_max_x, bounding_box_3D_max_y, bounding_box_3D_min_z))
                p4_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_min_x, bounding_box_3D_max_y, bounding_box_3D_min_z))

                p6_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_min_x, bounding_box_3D_min_y, bounding_box_3D_max_z))
                p7_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_max_x, bounding_box_3D_min_y, bounding_box_3D_max_z))
                p8_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_max_x, bounding_box_3D_max_y, bounding_box_3D_max_z))
                p5_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_min_x, bounding_box_3D_max_y, bounding_box_3D_max_z))

                p1_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p1_3DBB_glob)
                p2_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p2_3DBB_glob)
                p3_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p3_3DBB_glob)
                p4_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p4_3DBB_glob)
                p5_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p5_3DBB_glob)
                p6_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p6_3DBB_glob)
                p7_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p7_3DBB_glob)
                p8_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p8_3DBB_glob)

                p1_3DBB_pixel = [p1_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p1_3DBB_camera.y), p1_3DBB_camera.z]
                p2_3DBB_pixel = [p2_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p2_3DBB_camera.y), p2_3DBB_camera.z]
                p3_3DBB_pixel = [p3_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p3_3DBB_camera.y), p3_3DBB_camera.z]
                p4_3DBB_pixel = [p4_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p4_3DBB_camera.y), p4_3DBB_camera.z]
                p5_3DBB_pixel = [p5_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p5_3DBB_camera.y), p5_3DBB_camera.z]
                p6_3DBB_pixel = [p6_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p6_3DBB_camera.y), p6_3DBB_camera.z]
                p7_3DBB_pixel = [p7_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p7_3DBB_camera.y), p7_3DBB_camera.z]
                p8_3DBB_pixel = [p8_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p8_3DBB_camera.y), p8_3DBB_camera.z]

                # save 3D bounding box's edges as tuples of corners
                info_on_each_frame_for_each_worker['BB3D'] = [(p1_3DBB_pixel, p2_3DBB_pixel), (p2_3DBB_pixel, p3_3DBB_pixel), (p3_3DBB_pixel, p4_3DBB_pixel), (p4_3DBB_pixel, p5_3DBB_pixel), (p4_3DBB_pixel, p1_3DBB_pixel), (
                    p5_3DBB_pixel, p6_3DBB_pixel), (p6_3DBB_pixel, p7_3DBB_pixel), (p6_3DBB_pixel, p1_3DBB_pixel), (p7_3DBB_pixel, p8_3DBB_pixel), (p7_3DBB_pixel, p2_3DBB_pixel), (p8_3DBB_pixel, p3_3DBB_pixel), (p8_3DBB_pixel, p5_3DBB_pixel)]

            # writing the data to the dictionaries
            info_on_each_frame[rig_name] = info_on_each_frame_for_each_worker
            logger.debug(f'Ground truths for {rig_name} are extracted')
            logger.debug(f'2D Bboxes for {rig_name} is {info_on_each_frame_for_each_worker["BB2D"]}')
            logger.debug(f'3D Bboxes for {rig_name} is {info_on_each_frame_for_each_worker["BB3D"]}')
            logger.debug(f'3D Bboxes in global coordinate for {rig_name} is {info_on_each_frame_for_each_worker["BB3D_global_coordinate"]}')
            logger.debug(f'3D bone locations for {rig_name} is {info_on_each_frame_for_each_worker["bone_location_3d"]}')
            logger.debug(f'2D bone locations for {rig_name} is {info_on_each_frame_for_each_worker["bone_location_2d"]}')

        Information_dict[str(frame)] = info_on_each_frame
        Information_dict['lighting'] = lighting
        Information_dict['workers_name_list'] = workers_name_list

    # Setting the directory for saving dictionaries
    # os.chdir(path)

    logger.debug(f'Saving pickle to {path + "/Joint_Tracker.pickle"}')
    # Saving dictionaries
    with open(os.path.join(path,'Joint_Tracker.pickle'), 'wb') as handle:
        pickle.dump(Information_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



"Depth Map Function"
def Depth_Map_Genrator(output_path):
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_mist = True
    bpy.context.scene.world.mist_settings.start = 0
    bpy.context.scene.world.mist_settings.depth = 25

    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear the previously created nodes
    for node in tree.nodes:
        if node.name == "Depth_Maper" or node.name == "Invertor" or node.name == "Depth_Map_Output":
            tree.nodes.remove(node)

    rl = tree.nodes['Render Layers']

    # creating and adjusting Output node's settings
    output = tree.nodes.new(type="CompositorNodeOutputFile")
    output.name = "Depth_Map_Output"
    output.base_path = output_path
    output.format.color_mode = 'BW'
    #FIXME: set the color depth to 16
    # import ipdb; ipdb.set_trace()
    # output.format.color_depth = '16'
    output.location = [420, 230]

    # linking the render layer with output
    links.new(rl.outputs['Mist'], output.inputs['Image'])



"Segmantation function"
def Setup_Segmentation(output_path, occlusion_thrsh=0.98):

    bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True

    workers = [worker for worker in bpy.data.objects if 'Armature' in worker.name]
    workers = sorted(workers, key=lambda x: x.name)     # sort the workers based on their name to make masks consistant
    
    index = 254
    # indexing the worker object in scene
    for worker in workers:
        for obj in bpy.data.objects:
            if (obj.parent == worker and obj.type == 'MESH'):
                obj.pass_index = index
        index -= 10
    
    background_index = 50
    
    for obj in bpy.data.objects:
        if obj.type =='MESH':
            if all(term not in obj.name for term in ['Floor', 'Horizon', 'Amrature']):
                if obj.parent:
                    if 'Armature' not in obj.parent.name:
                        obj.pass_index = background_index
                else:
                    obj.pass_index = background_index
            elif 'Horizon' in obj.name:
                obj.pass_index = 0 
            else:
                obj.pass_index = background_index

    # creating the required nodes for segmentation
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # clear the previously created nodes
    for node in tree.nodes:
        if node.name == "Segmentation_Calculator" or node.name == 'Segmentation_Output':
            tree.nodes.remove(node)

    # definition of Segmentation_Calculator
    convertor = tree.nodes.new(type="CompositorNodeMath")
    convertor.name = "Segmentation_Calculator"
    convertor.operation = 'DIVIDE'
    convertor.inputs[1].default_value = 255
    convertor.location = [220, 0]

    # linking the Segmentation_Calculator node to the Render Layers
    links = tree.links
    link = links.new(
        tree.nodes['Render Layers'].outputs['IndexOB'], convertor.inputs[0])

    # creating and adjusting Output node's settings
    output = tree.nodes.new(type="CompositorNodeOutputFile")
    output.name = 'Segmentation_Output'
    output.base_path = output_path
    output.format.color_mode = 'BW'
    output.location = [420, 0]

    # linking the Ouput node to the Render Layers and Math
    link2 = links.new(convertor.outputs['Value'], output.inputs['Image'])
    
    

"Render Engine Settings"


def render_setting(scene=bpy.context.scene):
    bpy.context.scene.cycles.max_bounces = int(config['max_bounces'])
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.samples = int(config['samples'])
    bpy.context.scene.cycles.time_limit = 1
    bpy.context.scene.cycles.tile_size = int(config['tile_size'])
    bpy.context.scene.cycles.adaptive_threshold = threshold
    bpy.context.scene.render.resolution_x = int(config['resolution_x'])
    bpy.context.scene.render.resolution_y = int(config['resolution_y'])
    bpy.context.scene.render.fps = framerate
    bpy.context.scene.frame_step = framerate
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.frame_start = 5


"Rendering Function"

def rendering_random_camera(lighting, itr, camera, target_rig, scene_name, workers_name_list, name_tag):
    # find root bone
    random_bone = random.choice(list(target_rig.data.bones))
    parent = random_bone.parent
    if parent is None:
        root_bone = random_bone
    else:
        while parent is not None:
            root_bone = parent
            parent = parent.parent

    # track target rig by camera
    camera.constraints.new(type="TRACK_TO")
    camera.constraints['Track To'].target = target_rig
    camera.constraints['Track To'].subtarget = root_bone.name

    # remove previous cam_circles from scene
    for ob in bpy.data.objects.values():
        ob.select_set(False)
        try:
            for object in bpy.data.objects:
                if 'cam_circle' in object.name:
                    object.select_set(True)
                    if object.select_get():
                        bpy.ops.object.delete(use_global=False, confirm=False)
        except:
            pass
    
    # check occlusion and replace camera
    occlusion_percentage = occlusion_detector(target_rig, camera)

    if occlusion_percentage>1:
        logger.info('Camera is occluded, replacing the camera')
        perturb_orientation(camera_record, target_rig.name, target_rig)

    # set the start and end of the render to the target rig's animation start and finish
    keyframes = []
    anim = target_rig.animation_data
    if anim is not None and anim.action is not None:
        for fcu in anim.action.fcurves:
            for keyframe in fcu.keyframe_points:
                x, y = keyframe.co
                if x not in keyframes:
                    keyframes.append((math.ceil(x)))
    bpy.context.scene.frame_start = min(keyframes)
    bpy.context.scene.frame_end = max(keyframes)

    # creating name tag
    # name_tag =  'TC' + '_' + str(itr) + '_' + scene_name + '_' + target_rig.name.replace('Armature: ', '')
    name_tag = name_tag.replace('TC_', f'TC_{itr}_')
    if WORKER_ID is not None:
        name_tag = f"W{WORKER_ID:02d}_" + name_tag
    # creating new folder for new camera location
    os.chdir(Dataset_dir)
    # import ipdb; ipdb.set_trace()
    os.makedirs(str(name_tag), exist_ok= True)
    os.chdir(os.path.join(Dataset_dir, str(name_tag)))
    bpy.context.scene.render.filepath = os.path.join(os.getcwd(), 'test')

    # set the render properties
    render_setting()

    # calling semantic segmentation function
    Setup_Segmentation(os.path.join(os.getcwd(), 'Semantic Segmentation'))

    # calling depth map function
    Depth_Map_Genrator(os.path.join(os.getcwd(), 'Depth Map'))

    # rendering the animation
    # import ipdb; ipdb.set_trace()
    bpy.ops.render.render(animation=True, write_still=True)

    # print('Capturing scene information\n****************************************************\n')
    # calling data capture function
    joint_tracker(lighting, workers_name_list, path=os.getcwd())
    logger.info('Rendered for ' + target_rig.name + ' in ' + scene_name + ' scene.')




"""
Main Body
"""

# with open(report_path, 'w') as f:
logger.info(f"Starting data generation loop for {Number_of_Image_Sequences} sequences. "
            f"Dataset_dir={Dataset_dir}, logs_dir={logs_dir}, worker_id={WORKER_ID}")
scenepath = scene_path
scene_name = scenepath.name.replace('.blend', '')

"Append horizon"
with bpy.data.libraries.load(horizon_path.as_posix()) as (data_from, data_to):
    data_to.objects = [name for name in data_from.objects]

# link them to scene
scene = bpy.context.scene
for i, obj in enumerate(data_to.objects):
    if obj is not None:
        scene.collection.objects.link(obj)
        obj.name = f'Horizon{i+1}'
logger.info('Horizon is loaded')


"Append scene"
with bpy.data.libraries.load(scenepath.as_posix()) as (data_from, data_to):
    data_to.objects = [name for name in data_from.objects if (
        'Camera' and 'Light' and 'Plane') not in name]

# link them to scene
scene = bpy.context.scene
for obj in data_to.objects:
    if obj is not None: 
        scene.collection.objects.link(obj)

# Snap scene to the ground
scene = [obj for obj in bpy.data.objects if 'Horizon' not in obj.name]

# find the lowest Z value of scene
scene_lowest_pt = 1 * float('inf')
for mesh in scene:
    if mesh.type == 'MESH':
        # get the minimum z-value of all vertices after converting to global transform
        mesh_lowest_pt = min(
            [(mesh.matrix_world @ v.co).z for v in mesh.data.vertices])
        if mesh_lowest_pt < scene_lowest_pt:
            scene_lowest_pt = mesh_lowest_pt

# snap the scene to the ground
for obj in bpy.data.objects:
    if 'Horizon' not in obj.name:
        obj.location.z -= scene_lowest_pt        
    if 'Horizon' in obj.name:
        obj.location.z -= 0.1 # create a buffer between the scene and the horizon
logger.info('Scene Loaded:' + str(scenepath))



"Generate sky texture"
# remove any previous sky_texture

# add new sky texture
sky_texture = bpy.context.scene.world.node_tree.nodes.new(
    "ShaderNodeTexSky")
bg = bpy.context.scene.world.node_tree.nodes["Background"]
bpy.context.scene.world.node_tree.links.new(
    bg.inputs["Color"], sky_texture.outputs["Color"])
sky_texture.sky_type = 'NISHITA'

bpy.data.worlds['World'].node_tree.nodes['Sky Texture'].sun_intensity = 0.1
bpy.data.worlds['World'].node_tree.nodes['Sky Texture'].air_density = sun_state * 10
bpy.data.worlds['World'].node_tree.nodes['Sky Texture'].sun_elevation = sun_elevation * 1.57

lighting = {}
lighting['sun_state'] = sun_state * 10
lighting['sun_elevation'] = sun_elevation * 1.57
logger.info('Sky texture is generated with sun state: ' + str(sun_state * 10) + ' and sun elevation: ' + str(sun_elevation * 1.57))


"Set the camera location based on record"
(_,target_avatar_location), = avatars_locations[target_rig_name].items()
retrieved_actual_target_avatar_location = normalized2actual(Vector(target_avatar_location), bpy.data.objects['Floor'])
logger.info('Target avatar location is set to: ' + str(retrieved_actual_target_avatar_location))

"Append avatars to scene"

logger.info('Loading Avatars:')
# import ipdb; ipdb.set_trace()
# try:
if len(avatar_paths) == 0:
    raise ValueError ('No avatars found in the directory')
elif len(avatar_paths) != Number_of_Workers:
    target_av_dir = Avatar_dir/f'{target_rig_name}.blend'
    if target_av_dir.exists():
        Number_of_Workers = len(avatar_paths)

for i in range(Number_of_Workers):
    filepath = avatar_paths[i]
    worker_name = filepath.name.replace('.blend', '')
    logger.info(f'Avatar {i}: {worker_name}')
    with bpy.data.libraries.load(filepath.as_posix()) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects if (
            'Camera' and 'Light' and 'Plane') not in name]

    # link them to scene
    scene = bpy.context.scene
    for obj in data_to.objects:
        if obj is not None:
            scene.collection.objects.link(obj)
        if 'Armature' in obj.name:
            obj.name = obj.name + ': ' + worker_name
    
    # set the origin of armature to the lowest point
    rig = bpy.data.objects[obj.name]
    # find the lowest Z value of rig
    rig_lowest_pt = 1*float('inf')

    for obj in bpy.data.objects:
        if (obj.parent == rig and obj.type == 'MESH'):

            # get the minimum z-value of all vertices after converting to global transform
            mesh_lowest_pt = min(
                [(obj.matrix_world @ v.co).z for v in obj.data.vertices])

            if mesh_lowest_pt < rig_lowest_pt:
                rig_lowest_pt = mesh_lowest_pt
    rig.location.z += rig.location.z - rig_lowest_pt
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
    

"Localize avatars in scene"
rigs = [obj for obj in bpy.data.objects if 'Armature' in obj.name]
for rig in rigs:
    for worker in config['worker_bones_info']:
        if worker in rig.name:
            # find root bone
            random_bone = random.choice(list(rig.data.bones))
            parent = random_bone.parent
            if parent is None:
                root_bone = random_bone
            else:
                while parent is not None:
                    root_bone = parent
                    parent = parent.parent
            for key, value in config['worker_bones_info'][worker].items():
                hip_one_name = key
            arm_current_hip =  rig.matrix_world @ rig.data.bones[root_bone.name].head
            retrieved_normalized_hip_location = Vector(config['worker_bones_info'][worker][hip_one_name])
            retrieved_actual_hip_location = normalized2actual(retrieved_normalized_hip_location, bpy.data.objects['Floor'])
            nearest_workzone = get_nearest_workzone(bpy.data.objects['Floor'], retrieved_actual_hip_location)
            # print(retrieved_normalized_hip_location, retrieved_actual_hip_location, nearest_workzone)
            if nearest_workzone:
                rig.location = nearest_workzone
            else:
                transition = retrieved_actual_hip_location - arm_current_hip
                rig.location += transition
    if target_rig_name in rig.name:
        target_rig = rig
logger.info('Avatars are loaded')
    
itr = 0
for i in range(Number_of_Image_Sequences):
    itr += 1
    "Add camera to the scene"
    new_camera(focal_len=random.choice([15, 20, 30, 40]))
    camera = bpy.data.objects['Camera']
    # import ipdb; ipdb.set_trace()
    # localize_camera_based_on_record(camera_record, avatar_name=target_rig.name)
    perturb_orientation(camera_record, target_rig.name, target_rig)

    
    # select random rig as target rig
    workers_name_list = [rig.name.replace('Armature: ', '') for rig in rigs]
    
    # render the scene
    name_tag = timestamp + '_TC_' + \
        scene_name + '_' + \
        target_rig.name.replace('Armature: ', '')
    

        
    try:
        logger.info(f'Rendering random camera for the {itr}rd time with avatar {target_rig.name} in {scene_name} scene.')
        # render the scene
        rendering_random_camera(
            lighting=lighting, 
            workers_name_list=workers_name_list, 
            itr= itr, 
            camera=camera, 
            target_rig=target_rig, 
            scene_name=scene_name,
            name_tag=name_tag)
        logger.info(f'Rendering done for {str(name_tag)}')
    except Exception as e:
        logger.error(f'Error in rendering {str(name_tag)}: {str(e)}')

