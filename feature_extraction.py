import torch
from torchvision.models import resnet101, densenet201
import torch.nn as nn
import argparse
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sdal_utils.sdal_utils import load_image_paths_from_folder, extract_and_save_features_batchwise #, extract_and_save_features_batchwise_Dinov2
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from sdal_utils.sdal_utils import CustomImageDataset, DataLoader
import h5py
from tqdm import tqdm

def extract_features_from_images_DenseNet(data_directory, batch_size, hdf5_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = densenet201(pretrained=True)
    model.classifier = nn.Identity()
    model.eval()
    model = model.to(device)

    image_paths = load_image_paths_from_folder(data_directory)
    extract_and_save_features_batchwise(image_paths, model, transform, device, batch_size, hdf5_path)

    print(f'{len(image_paths)} number of images are converted')

def extract_features_from_images_Dinov2(data_directory, batch_size, hdf5_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
    model.eval()

    image_paths = load_image_paths_from_folder(data_directory)
    dataset = CustomImageDataset(image_paths, transform=processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with h5py.File(hdf5_path, 'w') as h5f:
        feature_shape = (768,)
        h5f.create_dataset('image_paths', data=image_paths, dtype=h5py.string_dtype(), compression="gzip")
        features_dset = h5f.create_dataset('features', shape=(len(image_paths), *feature_shape), dtype='float32', compression="gzip")

        for i, images_batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Extracting Features"):
            inputs = processor(images=images_batch, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                features_batch = outputs.last_hidden_state
                features_batch = features_batch.reshape(features_batch.size(0), -1)

            start_idx = i * batch_size
            end_idx = start_idx + features_batch.shape[0]
            features_dset[start_idx:end_idx] = features_batch.cpu().numpy()

    print(f'{len(image_paths)} number of images are converted')

def main():
    parser = argparse.ArgumentParser(description="Feature extraction from images")
    parser.add_argument('model', choices=['densenet', 'dinov2'], help="Model to use for feature extraction")
    parser.add_argument('data_directory', type=str, help="Directory containing the images")
    parser.add_argument('batch_size', type=int, help="Batch size for feature extraction")
    parser.add_argument('hdf5_path', type=str, help="Path to save the extracted features in HDF5 format")

    args = parser.parse_args()

    if args.model == 'densenet':
        extract_features_from_images_DenseNet(args.data_directory, args.batch_size, args.hdf5_path)
    elif args.model == 'dinov2':
        extract_features_from_images_Dinov2(args.data_directory, args.batch_size, args.hdf5_path)

if __name__ == "__main__":
    main()

# Example usage:
# python feature_extraction.py densenet /path/to/synthetic/dataset 2048 ./features/features_DenseNet.hdf5