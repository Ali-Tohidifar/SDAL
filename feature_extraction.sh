#!/bin/bash

# Simple feature extraction script that works on any machine

# Default parameters (change these as needed)
MODEL="densenet"
# Path to your dataset directory - MODIFY THIS
DATA_DIR="/path/to/your/synthetic/dataset"
BATCH_SIZE=2048
OUTPUT_DIR="./features"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Generate output filename with model name
OUTPUT_FILE="${OUTPUT_DIR}/features_${MODEL}.hdf5"

# Run feature extraction
echo "Extracting features using $MODEL..."
echo "Data directory: $DATA_DIR"
echo "Output file: $OUTPUT_FILE"

python feature_extraction.py $MODEL $DATA_DIR $BATCH_SIZE $OUTPUT_FILE

echo "Feature extraction complete!"