import bpy
import os
import csv
import matplotlib.pyplot as plt

def extract_sizes_and_plot(blender_files_dir, output_csv, output_image):
    all_volumes = []

    for file_name in os.listdir(blender_files_dir):
        if file_name.endswith('.blend'):
            file_path = os.path.join(blender_files_dir, file_name)
            
            # Open the blend file
            bpy.ops.wm.open_mainfile(filepath=file_path)
            
            if "Floor" in bpy.data.objects:
                floor = bpy.data.objects["Floor"]
                scale = floor.scale
                dimensions = floor.dimensions
                h,v = sorted(floor.dimensions, reverse=True)[0], sorted(floor.dimensions, reverse=True)[1] 
                # volume = dimensions[0] * scale[0] * dimensions[1] * scale[1] * dimensions[2] * scale[2]
                all_volumes.append((h*v, h, v))
            else:
                print(f"No object named 'Floor' found in {file_name}.")
    
    # Save data to CSV
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File', 'َArea', 'Height', 'Width'])
        for file_name, volume in zip(os.listdir(blender_files_dir), all_volumes):
            if file_name.endswith('.blend'):
                writer.writerow([file_name, volume[0], volume[1], volume[2]])
    
    # Generate histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_volumes[1], bins=20, color='blue', alpha=0.7)
    plt.title('Histogram of Floor Object Volumes')
    plt.xlabel('Area')
    plt.ylabel('Frequency')
    plt.savefig(output_image)
    plt.close()

# Define paths
blender_files_directory = r"E:\Data_Generator_WorkerDetection\3DAssets_GCPBucket\Scenes"
output_csv_file = 'output_sizes.csv'
output_jpg_file = 'size_histogram.jpg'

# Run the function
extract_sizes_and_plot(blender_files_directory, output_csv_file, output_jpg_file)
