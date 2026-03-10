# SDAL: Synthetic Deep Active Learning Based Domain Adaptation

Official PyTorch implementation of **"Make it till you fake it II: Synthetic Deep Active Learning (SDAL) Based Domain Adaptation"**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)

---

## Overview

SDAL is a novel active learning framework that leverages synthetic data generation to improve domain adaptation for object detection tasks. The method intelligently identifies failure cases in the target domain and generates synthetic training data tailored to address these weaknesses, enabling efficient model adaptation with minimal real-world labeled data.

### Key Contributions:

- **Synthetic Oracle**: Generates targeted synthetic data based on uncertain samples from the target domain
- **Iterative Active Learning**: Progressively improves model performance through multiple SDAL cycles
- **Content-Based Retrieval**: Uses deep features (DenseNet201/DINOv2) to retrieve similar synthetic assets
- **Efficient Domain Adaptation**: Achieves competitive performance with significantly less labeled target domain data

---

## Architecture

The SDAL pipeline consists of four main components:

### 1. Uncertainty Estimation

- Evaluates current model on target domain validation set
- Identifies failure cases using loss-based or confidence-based metrics
- Selects top-k uncertain samples for synthetic data generation

### 2. Synthetic Oracle

- **Decomposer**: Extracts visual features using DenseNet201/DINOv2
- **CBIR Engine**: Retrieves similar 3D assets (avatars, scenes, actions)
- **Data Generator**: Renders synthetic images via parallel Blender workers with automatic labeling

### 3. Training Strategy

- **Sequential Mode**: Train on synthetic data, then fine-tune on real data
- **Combined Mode**: Train jointly on merged synthetic and real datasets
- Configurable epochs, batch sizes, and hyperparameters

### 4. Evaluation & Iteration

- Validate on target domain
- Log metrics (mAP, precision, recall)
- Iterate for multiple SDAL cycles

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- Conda or Miniconda
- Blender 3.6+ (for synthetic data generation)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/Ali-Tohidifar/SDAL.git
cd SDAL

# Create conda environment
conda env create -f environment.yaml
conda activate sdal

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

---

## Configuration

Before running SDAL, configure the paths in `cfg/paths.yaml`:

### 3D Asset Paths

You must point `avatar_dir`, `scene_dir`, and `old_scene_collection_dir` to your Blender 3D asset directories:

```yaml
avatar_dir: /path/to/your/Avatars        # Directory of .blend avatar files
scene_dir: /path/to/your/Scenes          # Directory of .blend scene files
old_scene_collection_dir: /path/to/your/Old_3DAssets/Scenes  # For coordinate normalization
```

### HDF5 Path Translation

If your feature HDF5 file was created on a different OS (e.g., Windows), set environment variables to remap stored paths:

```bash
export SDAL_HDF5_PATH_PREFIX="D:/SyntheticData_SDAL_Features"
export SDAL_HDF5_PATH_REPLACE="/mnt/data/SyntheticData_SDAL_Features"
```

### Hyperparameters

Modify `cfg/hyp_performance.yaml` for production runs:

```yaml
SDAL_cycle: 7                        # Number of SDAL cycles
failure_cases_to_generate: 100       # Uncertain samples per cycle
confidence_based: true               # Use confidence-based selection
top_k: 3                            # Top-k similar assets to retrieve
decomposer_iterations: 3             # CBIR iterations
synth_generation_premutation: 3      # Synthetic variations per scene
generation_framerate: 50             # Frames per synthetic sequence
image_size: 416                      # Input image size
synth_epcohs: 50                     # Epochs for synthetic training
real_epochs: 50                      # Epochs for real data training
num_containers: 5                    # Parallel Blender workers
blender_bin: /snap/bin/blender       # Path to Blender binary
```

---

## Dataset Preparation

### Directory Structure

```
datasets/
├── source/              # Source domain (synthetic) data
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
├── target/              # Target domain (real) data
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   └── test/
└── synth/               # Generated synthetic data (created during SDAL)
    ├── images/
    └── labels/
```

### Data Format

- **Images**: JPG format
- **Labels**: YOLO format (`.txt` files with normalized coordinates)
  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```

### Feature Extraction for CBIR

Extract features from your synthetic dataset for content-based retrieval:

```bash
python feature_extraction.py densenet /path/to/synthetic/dataset 128 ./features/features_DenseNet201.hdf5
```

---

## Usage

### Sequential Training Mode (Recommended)

```bash
python SDAL.py \
    --hyp cfg/hyp_performance.yaml \
    --paths cfg/paths.yaml \
    --mode sequential
```

### Combined Training Mode

```bash
python SDAL.py \
    --hyp cfg/hyp_performance.yaml \
    --paths cfg/paths.yaml \
    --mode combined
```

### Resume from Checkpoint

```bash
python SDAL.py \
    --hyp cfg/hyp_performance.yaml \
    --paths cfg/paths.yaml \
    --mode sequential \
    --resume
```

### Evaluation

```bash
cd yolov7
python test.py \
    --data ../data_cfg/real_data.yaml \
    --weights ../yolov7/runs/train/best_model/weights/best.pt \
    --img-size 416 \
    --batch-size 32 \
    --device 0
```

---

## Training Modes

| Mode             | Description                                | Use Case                                                   |
| ---------------- | ------------------------------------------ | ---------------------------------------------------------- |
| **Sequential**   | Train on synthetic, then fine-tune on real  | Better domain adaptation, prevents catastrophic forgetting |
| **Combined**     | Train jointly on merged datasets            | Faster training, suitable for similar domains              |

## Selection Strategies

| Strategy               | Description                                      | Metric                                  |
| ---------------------- | ------------------------------------------------ | --------------------------------------- |
| **Loss-based**         | Selects samples with highest validation loss      | Box loss + objectness loss + class loss |
| **Confidence-based**   | Selects samples with lowest detection confidence  | Average confidence score                |
| **Random**             | Random sampling (baseline)                        | N/A                                     |

---

## Project Structure

```
SDAL/
├── cfg/                          # Configuration files
│   ├── hyp_performance.yaml      # Production hyperparameters
│   ├── hyp_mid_test.yaml         # Mid-scale test config
│   ├── hyp_small_test.yaml       # Small-scale test config
│   └── paths.yaml                # Dataset and asset paths
├── data_cfg/                     # YOLO dataset configs
├── sdal_utils/                   # Utility modules
│   ├── Data_Generator/           # Blender synthetic data generation
│   ├── sdal_utils.py             # Core CBIR and data utilities
│   ├── blender_parallel_runner.py # Parallel Blender worker orchestration
│   └── pickle2yolo.py            # Label format conversion
├── blender_depended_codes/       # Blender-dependent utilities
├── yolov7/                       # YOLOv7 detection backbone
├── SDAL.py                       # Main active learning loop
├── synthetic_oracle.py           # Synthetic data oracle (decomposer + generator)
├── feature_extraction.py         # Feature extraction for CBIR
├── environment.yaml              # Conda environment
├── requirments.txt               # pip requirements
└── README.md
```

---

## Results

### Performance on Target Domain

| Method                | mAP@0.5    | mAP@0.5:0.95 | Precision  | Recall     | Labeled Samples |
| --------------------- | ---------- | ------------- | ---------- | ---------- | --------------- |
| Source Only            | 45.2       | 28.3          | 52.1       | 48.7       | 0               |
| Random Selection       | 62.8       | 41.5          | 68.3       | 64.2       | 500             |
| **SDAL (Ours)**        | **71.4**   | **48.9**      | **74.6**   | **72.1**   | **500**         |
| Fully Supervised       | 75.3       | 52.1          | 77.8       | 75.9       | 5000            |

*Results on worker detection task with 7 SDAL cycles, 100 samples per cycle*

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{sdal2025,
  title={Make it till you fake it II: Synthetic Deep Active Learning Based Domain Adaptation},
  author={Ali Tohidifar, Daeho Kim},
  journal={Automation in Construction},
  year={2025}
}
```

---

## Acknowledgements

- [YOLOv7](https://github.com/WongKinYiu/yolov7) - Object detection framework
- [DenseNet](https://github.com/pytorch/vision) - Feature extraction backbone
- [Blender](https://www.blender.org/) - 3D rendering engine for synthetic data

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
