#!/bin/bash

# Path to Python executable
# Use python from the activated conda environment
# If you need a specific python path, modify this variable
PYTHON_PATH="python"  # or specify: "/path/to/conda/envs/sdal/bin/python"

# Paths to the necessary configuration files
HYP_FILE="./cfg/hyp_mid_test.yaml"
PATHS_FILE="./cfg/paths.yaml"

# Warming up a model
# echo "Warming up a model..."
# python ./yolov7/train.py --project ./yolov7/runs/train --img-size 416 --batch 64 --epochs 100 --data data_cfg/warmup.yaml --weights "" --name warmup_Real_5K_416 --cfg cfg/training/yolov7-tiny.yaml --hyp data/hyp.scratch.tiny.yaml

# Run the SDAL script with the specified configuration files
echo "Running SDAL with the following configuration files:"
echo "Hyp: $HYP_FILE"
cat -n "$HYP_FILE"
echo
echo "Paths: $PATHS_FILE"
cat -n "$PATHS_FILE"
echo

"$PYTHON_PATH" SDAL.py --hyp "$HYP_FILE" --paths "$PATHS_FILE" --mode combined

echo "SDAL run exited."
