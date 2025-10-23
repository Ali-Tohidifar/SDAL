import cv2
import os
import numpy as np
import argparse

def read_yolo_labels(label_path):
    """Reads YOLO labels from a file."""
    with open(label_path, 'r') as file:
        labels = file.readlines()
    return labels

def resize_image_and_adjust_labels(image_path, label_path, output_image_path, output_label_path, target_size=(960, 960)):
    """Resizes the image, adjusts labels, and saves the resized image and adjusted labels."""
    image = cv2.imread(image_path)
    labels = read_yolo_labels(label_path)

    # Get original dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate the scaling factor
    scale = min(target_size[0] / original_width, target_size[1] / original_height)
    
    # Resize the image
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Create a new image with grey padding
    padded_image = np.full((target_size[1], target_size[0], 3), 128, dtype=np.uint8)  # 128 for grey padding
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    
    # Adjust bounding box coordinates
    adjusted_labels = []
    for label in labels:
        components = label.strip().split()
        class_id = components[0]
        bbox_x, bbox_y, bbox_w, bbox_h = map(float, components[1:])
        
        # Adjust coordinates according to new size
        bbox_x = (bbox_x * original_width * scale + x_offset) / target_size[0]
        bbox_y = (bbox_y * original_height * scale + y_offset) / target_size[1]
        bbox_w *= scale * original_width / target_size[0]
        bbox_h *= scale * original_height / target_size[1]

        adjusted_labels.append(f"{class_id} {bbox_x} {bbox_y} {bbox_w} {bbox_h}\n")
    
    # Save the resized image and adjusted labels
    cv2.imwrite(output_image_path, padded_image)
    with open(output_label_path, 'w') as file:
        file.writelines(adjusted_labels)

def process_dataset(dataset_dir, output_dataset_dir, target_size=(960, 960)):
    """Processes the entire dataset of images and labels."""
    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')
    output_image_dir = os.path.join(output_dataset_dir, 'images')
    output_label_dir = os.path.join(output_dataset_dir, 'labels')

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(image_name)[0]
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, base_name + '.txt')

            # Convert to JPG if necessary
            if not image_name.lower().endswith('.jpg'):
                image = cv2.imread(image_path)
                image_path = os.path.join(image_dir, base_name + '.jpg')
                cv2.imwrite(image_path, image)

            output_image_path = os.path.join(output_image_dir, base_name + '.jpg')
            output_label_path = os.path.join(output_label_dir, base_name + '.txt')

            resize_image_and_adjust_labels(image_path, label_path, output_image_path, output_label_path, target_size)

def main():
    parser = argparse.ArgumentParser(description="Resize images and adjust YOLO bounding boxes.")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory containing images and labels subdirectories.')
    parser.add_argument('--output_dataset_dir', type=str, required=True, help='Path to the output dataset directory to save resized images and adjusted labels.')
    parser.add_argument('--target_size', type=int, nargs=2, default=(960, 960), help='Target size for resizing images (width height).')
    
    args = parser.parse_args()

    process_dataset(args.dataset_dir, args.output_dataset_dir, tuple(args.target_size))

if __name__ == "__main__":
    main()
