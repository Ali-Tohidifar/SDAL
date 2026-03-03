#!/bin/bash
# Cleanup script for SDAL Mode-B experiments
# This script removes all generated data and experiment outputs to start fresh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "SDAL Mode-B Experiment Cleanup"
echo "=========================================="
echo ""

# Read num_containers from config if it exists
CONFIG_YAML="./cfg/modeB_hyp.yaml"
NUM_CONTAINERS=5  # Default
if [ -f "$CONFIG_YAML" ]; then
    NUM_CONTAINERS=$(grep -E "^num_containers:" "$CONFIG_YAML" | sed 's/.*num_containers:[[:space:]]*\([0-9]*\).*/\1/' || echo "5")
    if ! [[ "$NUM_CONTAINERS" =~ ^[0-9]+$ ]]; then
        NUM_CONTAINERS=5
    fi
fi

echo "Detected num_containers: $NUM_CONTAINERS"
echo ""

# 1. Remove all experiment runs
echo "1. Removing experiment output directories..."
EXPERIMENTS_DIR="./experiments/modeB_runs"
if [ -d "$EXPERIMENTS_DIR" ]; then
    rm -rf "$EXPERIMENTS_DIR"/*
    echo "   ✓ Cleared: $EXPERIMENTS_DIR"
else
    echo "   - Directory does not exist: $EXPERIMENTS_DIR"
fi

# 2. Remove Dataset directories (Dataset_1 through Dataset_N)
echo ""
echo "2. Removing generated synthetic data directories..."
for i in $(seq 1 $NUM_CONTAINERS); do
    DATASET_DIR="./sdal_utils/Data_Generator/Dataset_$i"
    if [ -d "$DATASET_DIR" ]; then
        rm -rf "$DATASET_DIR"/*
        echo "   ✓ Cleared: $DATASET_DIR"
    fi
done

# 3. Remove logs directories (logs_1 through logs_N)
echo ""
echo "3. Removing Blender generation logs..."
for i in $(seq 1 $NUM_CONTAINERS); do
    LOGS_DIR="./sdal_utils/Data_Generator/logs_$i"
    if [ -d "$LOGS_DIR" ]; then
        rm -rf "$LOGS_DIR"/*
        echo "   ✓ Cleared: $LOGS_DIR"
    fi
done

# 4. Remove archived dataset directory
echo ""
echo "4. Removing archived dataset directory..."
DATASET_USED_DIR="./sdal_utils/Data_Generator/Dataset_used"
if [ -d "$DATASET_USED_DIR" ]; then
    rm -rf "$DATASET_USED_DIR"/*
    echo "   ✓ Cleared: $DATASET_USED_DIR"
else
    echo "   - Directory does not exist: $DATASET_USED_DIR"
fi

# 5. Remove experiment logs
echo ""
echo "5. Removing experiment log files..."
LOG_DIR="./logs/modeB"
if [ -d "$LOG_DIR" ]; then
    rm -f "$LOG_DIR"/*.log
    echo "   ✓ Cleared: $LOG_DIR/*.log"
else
    echo "   - Directory does not exist: $LOG_DIR"
fi

# 6. Remove config.yaml (will be regenerated)
echo ""
echo "6. Removing generated config.yaml..."
CONFIG_YAML="./sdal_utils/Data_Generator/config.yaml"
if [ -f "$CONFIG_YAML" ]; then
    rm -f "$CONFIG_YAML"
    echo "   ✓ Removed: $CONFIG_YAML"
else
    echo "   - File does not exist: $CONFIG_YAML"
fi

# 7. Optional: Clean Docker containers and images
echo ""
read -p "7. Remove Docker containers and images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Removing Docker containers..."
    containers=$(docker ps -a -q --filter "ancestor=adaptive_blendcon" 2>/dev/null || true)
    if [ -n "$containers" ]; then
        docker rm -f $containers 2>/dev/null || true
        echo "   ✓ Removed Docker containers"
    else
        echo "   - No containers found"
    fi
    
    echo "   Removing Docker images..."
    images=$(docker images -q adaptive_blendcon 2>/dev/null || true)
    if [ -n "$images" ]; then
        docker rmi -f $images 2>/dev/null || true
        echo "   ✓ Removed Docker images"
    else
        echo "   - No images found"
    fi
else
    echo "   - Skipped Docker cleanup"
fi

# 8. Remove cache files from dataset directories (if any exist)
echo ""
echo "8. Removing cache files..."
find ./sdal_utils/Data_Generator -name "*.cache" -type f -delete 2>/dev/null || true
find ./experiments -name "*.cache" -type f -delete 2>/dev/null || true
find ./datasets -name "*.cache" -type f -delete 2>/dev/null || true
echo "   ✓ Removed cache files"

echo ""
echo "=========================================="
echo "Cleanup complete! Ready for fresh experiment."
echo "=========================================="
