# SDAL: Synthetic Deep Active Learning Based Domain Adaptation

Official PyTorch implementation of **"Make it till you fake it II: Synthetic Deep Active Learning (SDAL) Based Domain Adaptation"**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Feature Extraction](#feature-extraction)
  - [Training SDAL](#training-sdal)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

SDAL is a novel active learning framework that leverages synthetic data generation to improve domain adaptation for object detection tasks. The method intelligently identifies failure cases in the target domain and generates synthetic training data tailored to address these weaknesses, enabling efficient model adaptation with minimal real-world labeled data.

<div align="center">
  <img src="assets/overview.png" alt="SDAL Overview" width="800"/>
</div>

### Key Contributions:

- **Synthetic Oracle**: Generates targeted synthetic data based on uncertain samples from the target domain
- **Iterative Active Learning**: Progressively improves model performance through multiple SDAL cycles
- **Content-Based Retrieval**: Uses deep features (DenseNet201/DINOv2) to retrieve similar synthetic assets
- **Efficient Domain Adaptation**: Achieves competitive performance with significantly less labeled target domain data

---

## Key Features

- **Iterative Active Learning Pipeline**: Multi-cycle training strategy with synthetic data augmentation
- **Synthetic Data Generation**: Automated generation of labeled synthetic images using 3D rendering
- **Uncertainty-Based Selection**: Multiple selection strategies (loss-based, confidence-based, random)
- **Flexible Architecture**: Supports both sequential and combined training modes
- **Feature Extraction**: DenseNet201 and DINOv2 backbone support for CBIR
- **YOLOv7 Integration**: State-of-the-art object detection model
- **Checkpointing**: Resume training from interruptions
- **Comprehensive Logging**: Detailed tracking with WandB integration

---

## Architecture

The SDAL pipeline consists of four main components:

### 1. **Uncertainty Estimation**

- Evaluates current model on target domain validation set
- Identifies failure cases using loss-based or confidence-based metrics
- Selects top-k uncertain samples for synthetic data generation

### 2. **Synthetic Oracle**

- **Decomposer**: Extracts visual features using DenseNet201/DINOv2
- **CBIR Engine**: Retrieves similar 3D assets (avatars, scenes, actions)
- **Data Generator**: Renders synthetic images with automatic labeling

### 3. **Training Strategy**

- **Sequential Mode**: Train on synthetic data, then fine-tune on real data
- **Combined Mode**: Train jointly on merged synthetic and real datasets
- Configurable epochs, batch sizes, and hyperparameters

### 4. **Evaluation & Iteration**

- Validate on target domain
- Log metrics (mAP, precision, recall)
- Iterate for multiple SDAL cycles

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- Conda or Miniconda
- Docker (for synthetic data generation with Blender)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/SDAL.git
cd SDAL

# Create conda environment
conda env create -f environment.yaml
conda activate sdal

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Setup data generation assets (requires symlinks to 3D assets)
bash setup_linked_folders.sh
```

### Docker Setup for Synthetic Data Generation

```bash
cd sdal_utils/Data_Generator
docker build -t blendcon:latest .
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

### 1. Configuration

Create or modify configuration files in `cfg/`:

**`cfg/paths.yaml`** - Dataset and model paths:

```yaml
real_data_yaml: ./data_cfg/real_data.yaml
synth_data_yaml: ./data_cfg/synth_data.yaml
warmed_up_model_weights: ./warmed_up_models/weights/best.pt
hdf5_path: ./features/features_DenseNet201.hdf5
avatar_dir: ./sdal_utils/Data_Generator/Avatars
scene_dir: ./sdal_utils/Data_Generator/Scenes
log_dir: ./logs
```

**`cfg/hyp_performance.yaml`** - Hyperparameters:

```yaml
SDAL_cycle: 7                        # Number of SDAL cycles
failure_cases_to_generate: 100       # Uncertain samples per cycle
confidence_based: true               # Use confidence-based selection
random_selection: false              # Use random selection (for baseline)
top_k: 3                            # Top-k similar assets to retrieve
decomposer_iterations: 3             # CBIR iterations
synth_generation_premutation: 3      # Synthetic variations per scene
generation_framerate: 50             # Frames per synthetic sequence
image_size: 416                      # Input image size
synth_epcohs: 50                     # Epochs for synthetic training
real_epochs: 50                      # Epochs for real data training
device: 0                            # GPU device ID
```

### 2. Feature Extraction

Extract features from your synthetic dataset:

```bash
python feature_extraction.py densenet \
    /path/to/synthetic/dataset \
    128 \
    ./features/features_DenseNet201.hdf5
```

Or using the shell script:

```bash
bash feature_extraction.sh
```

### 3. Training SDAL

#### Sequential Training Mode (Recommended)

```bash
python SDAL.py \
    --hyp cfg/hyp_performance.yaml \
    --paths cfg/paths.yaml \
    --mode sequential
```

#### Combined Training Mode

```bash
python SDAL.py \
    --hyp cfg/hyp_performance.yaml \
    --paths cfg/paths.yaml \
    --mode combined
```

#### Resume from Checkpoint

```bash
python SDAL.py \
    --hyp cfg/hyp_performance.yaml \
    --paths cfg/paths.yaml \
    --mode sequential \
    --resume
```

### 4. Evaluation

Evaluate the trained model on the test set:

```bash
cd yolov7
python test.py \
    --data ../data_cfg/real_data.yaml \
    --weights ../yolov7/runs/train/best_model/weights/best.pt \
    --img-size 416 \
    --batch-size 32 \
    --device 0
```

### 5. Baseline Comparison

Train with random selection (no active learning):

```bash
# Set random_selection: true in hyp_performance.yaml
python SDAL.py \
    --hyp cfg/hyp_performance.yaml \
    --paths cfg/paths.yaml \
    --mode sequential
```

---

## Configuration

### Training Modes

| Mode                 | Description                                | Use Case                                                   |
| -------------------- | ------------------------------------------ | ---------------------------------------------------------- |
| **Sequential** | Train on synthetic, then fine-tune on real | Better domain adaptation, prevents catastrophic forgetting |
| **Combined**   | Train jointly on merged datasets           | Faster training, suitable for similar domains              |

### Selection Strategies

| Strategy                   | Description                                      | Metric                                  |
| -------------------------- | ------------------------------------------------ | --------------------------------------- |
| **Loss-based**       | Selects samples with highest validation loss     | Box loss + objectness loss + class loss |
| **Confidence-based** | Selects samples with lowest detection confidence | Average confidence score                |
| **Random**           | Random sampling (baseline)                       | N/A                                     |

### Hyperparameter Recommendations

| Task           | image_size | synth_epochs | real_epochs | failure_cases |
| -------------- | ---------- | ------------ | ----------- | ------------- |
| Small objects  | 640        | 75           | 75          | 150           |
| Medium objects | 416        | 50           | 50          | 100           |
| Large objects  | 320        | 30           | 30          | 50            |

---

## Project Structure

```
SDAL/
├── cfg/                          # Configuration files
│   ├── hyp_performance.yaml      # Hyperparameters
│   ├── hyp_mid_test.yaml         # Mid-scale test config
│   ├── hyp_small_test.yaml       # Small-scale test config
│   └── paths.yaml                # Dataset paths
├── data_cfg/                     # YOLO dataset configs
│   ├── real_data.yaml
│   └── synth_data.yaml
├── datasets/                     # Dataset directory
│   ├── source/
│   ├── target/
│   └── synth/
├── features/                     # Extracted features for CBIR
│   └── features_DenseNet201.hdf5
├── logs/                         # Training logs
├── sdal_utils/                   # Utility modules
│   ├── Data_Generator/           # Synthetic data generation
│   ├── sdal_utils.py             # Core utilities
│   ├── pickle2yolo.py            # Label conversion
│   └── DataComparison/           # Evaluation metrics
├── yolov7/                       # YOLOv7 submodule
├── warmed_up_models/             # Pre-trained weights
├── SDAL.py                       # Main training script
├── synthetic_oracle.py           # Synthetic data oracle
├── feature_extraction.py         # Feature extraction
├── environment.yaml              # Conda environment
├── requirements.txt              # pip requirements
└── README.md
```

---

## Results

### Performance on Target Domain

| Method                | mAP@0.5        | mAP@0.5:0.95   | Precision      | Recall         | Labeled Samples |
| --------------------- | -------------- | -------------- | -------------- | -------------- | --------------- |
| Source Only           | 45.2           | 28.3           | 52.1           | 48.7           | 0               |
| Random Selection      | 62.8           | 41.5           | 68.3           | 64.2           | 500             |
| **SDAL (Ours)** | **71.4** | **48.9** | **74.6** | **72.1** | **500**   |
| Fully Supervised      | 75.3           | 52.1           | 77.8           | 75.9           | 5000            |

*Results on worker detection task with 7 SDAL cycles, 100 samples per cycle*

### Selection Strategy Comparison

| Strategy                       | Overlap with Loss-based |
| ------------------------------ | ----------------------- |
| Confidence-based (100 samples) | 23%                     |
| Confidence-based (250 samples) | 66.8%                   |
| Confidence-based (500 samples) | 90.6%                   |

### Key Findings

- ✅ SDAL achieves 94.8% of fully supervised performance with only 10% labeled data
- ✅ Uncertainty-based selection outperforms random selection by +8.6 mAP@0.5
- ✅ Sequential training mode provides better domain adaptation than combined mode
- ✅ Performance plateaus after 5-7 cycles depending on domain complexity

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{sdal2024,
  title={Make it till you fake it II: Synthetic Deep Active Learning Based Domain Adaptation},
  author={Ali Tohidifar, Daeho Kim},
  booktitle={Automation in Construction},
  year={2025}
}
```

---

## Acknowledgements

This project builds upon several excellent open-source projects:

- [YOLOv7](https://github.com/WongKinYiu/yolov7) - Object detection framework
- [DenseNet](https://github.com/pytorch/vision) - Feature extraction backbone
- [Blender](https://www.blender.org/) - 3D rendering engine for synthetic data

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Issues & Contributing

If you encounter any issues or have suggestions for improvements, please open an issue on GitHub. Contributions are welcome!

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .
```

---


<div align="center">
  <sub>Built with ❤️ for advancing active learning and domain adaptation research</sub>
</div>
