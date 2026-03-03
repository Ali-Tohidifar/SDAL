#!/bin/bash
# Setup symlink from SDAL to SDAL+ datasets directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SDAL_PLUS_DATASETS="/home/ali_tohidi/SDAL+/datasets"
SDAL_DATASETS="./datasets"

echo "Setting up datasets symlink..."
echo "Source: $SDAL_PLUS_DATASETS"
echo "Link: $SDAL_DATASETS"
echo ""

# Check if source exists
if [ ! -d "$SDAL_PLUS_DATASETS" ]; then
    echo "ERROR: Source directory does not exist: $SDAL_PLUS_DATASETS"
    exit 1
fi

# Remove existing datasets if it exists (file, directory, or symlink)
if [ -e "$SDAL_DATASETS" ] || [ -L "$SDAL_DATASETS" ]; then
    echo "Removing existing datasets..."
    rm -rf "$SDAL_DATASETS"
fi

# Create symlink
echo "Creating symlink..."
ln -s "$SDAL_PLUS_DATASETS" "$SDAL_DATASETS"

# Verify symlink
if [ -L "$SDAL_DATASETS" ]; then
    echo "✓ Symlink created successfully"
    echo ""
    echo "Verifying symlink..."
    if [ -d "$SDAL_DATASETS/site_splits/lash_site_v1/modeB/day01/mine/images" ]; then
        echo "✓ Symlink works! Found day01/mine/images"
        echo ""
        echo "Symlink details:"
        ls -la "$SDAL_DATASETS" | head -3
    else
        echo "WARNING: Symlink created but path verification failed"
        echo "Please check if the source directory structure is correct"
    fi
else
    echo "ERROR: Failed to create symlink"
    exit 1
fi

echo ""
echo "Setup complete!"
