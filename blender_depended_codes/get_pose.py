import bpy
import argparse
import sys
from mathutils import Vector
import json

def calculate_normalized_position(obj_reference, target_world_location):
    """
    Calculate the normalized position based on a target world location with respect to the bounding box of obj_reference.
    
    Parameters:
        obj_reference (bpy.types.Object): The reference object whose bounding box is used.
        target_world_location (Vector): World location to normalize.

    Returns:
        Vector: The normalized relative position based on target_world_location within the bounding box of obj_reference.
    """
    # Update the scene, needed for getting updated location data
    bpy.context.view_layer.update()
    
    # Calculate bounding box dimensions
    bbox_corners = [obj_reference.matrix_world @ Vector(corner) for corner in obj_reference.bound_box]
    min_corner = Vector((min(c.x for c in bbox_corners), min(c.y for c in bbox_corners), min(c.z for c in bbox_corners)))
    max_corner = Vector((max(c.x for c in bbox_corners), max(c.y for c in bbox_corners), max(c.z for c in bbox_corners)))
    bbox_dimensions = max_corner - min_corner

    # Calculate relative position based on target world location
    relative_position = Vector(target_world_location) - min_corner

    # Normalize the relative position by the dimensions of the bounding box
    normalized_position = Vector((relative_position.x / bbox_dimensions.x if bbox_dimensions.x else 0,
                                  relative_position.y / bbox_dimensions.y if bbox_dimensions.y else 0,
                                  relative_position.z / bbox_dimensions.z if bbox_dimensions.z else 0))

    return normalized_position

def main():
    parser = argparse.ArgumentParser(description="Process points within a Blender environment.")
    parser.add_argument('--scene_dir', type=str, required=True, help='The .blend file to load.')
    parser.add_argument('--input_json', type=str, required=True, help='JSON file with points to process.')
    parser.add_argument('--output_file', type=str, default="output_vectors.json", help='Output JSON file for normalized vectors.')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    # Load the specified Blender file
    bpy.ops.wm.open_mainfile(filepath=args.scene_dir)

    with open(args.input_json, 'r') as f:
        points_data = json.load(f)

    floor = bpy.data.objects.get('Floor')
    if not floor:
        print("Error: Floor object not found.")
        return

    results = []
    for item in points_data:
        point = Vector(item['position'])
        normalized_pos = calculate_normalized_position(floor, point)
        results.append({
            "point": item['label'],
            "normalized_position": list(normalized_pos)
        })

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()