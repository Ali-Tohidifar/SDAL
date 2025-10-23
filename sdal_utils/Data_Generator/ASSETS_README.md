# 3D Assets Setup Guide

This directory requires large 3D assets (Blender files, avatars, scenes) that are not included in the Git repository due to their size.

## Required Assets

The SDAL pipeline requires the following assets:

### 1. Blender Scene Files
- **Empty.blend** - Empty Blender scene template
- **Horizon.blend** - Horizon environment scene (346 MB)

### 2. 3D Model Directories
- **Avatars/** - Directory containing avatar 3D models
- **Scenes/** - Directory containing scene 3D models  
- **Old_3DAssets/Scenes/** - Directory for legacy scene assets

## Setup Instructions

### Option 1: Download from Cloud Storage (Recommended)
If you have access to the project's cloud storage, download the assets:

```bash
# Example using your cloud storage
# Replace with your actual download method
cd sdal_utils/Data_Generator/
# Download and extract assets here
```

### Option 2: Use Your Own Assets
If you have your own 3D assets:

1. Create the required directories:
```bash
cd sdal_utils/Data_Generator/
mkdir -p Avatars Scenes Old_3DAssets/Scenes
```

2. Place your Blender scene files:
   - Add `Empty.blend` (empty scene template)
   - Add `Horizon.blend` (environment scene)

3. Add your 3D models:
   - Place avatar models in `Avatars/`
   - Place scene models in `Scenes/`
   - Place legacy assets in `Old_3DAssets/Scenes/`

### Option 3: Symbolic Links (For Local Development)
If assets are stored elsewhere on your system:

```bash
cd sdal_utils/Data_Generator/
ln -s /path/to/your/assets/Avatars Avatars
ln -s /path/to/your/assets/Scenes Scenes
ln -s /path/to/your/assets/Horizon.blend Horizon.blend
ln -s /path/to/your/assets/Empty.blend Empty.blend
```

## Required Directory Structure

After setup, your directory should look like:

```
Data_Generator/
├── ASSETS_README.md (this file)
├── Empty.blend              # Empty scene (not in git)
├── Horizon.blend            # Horizon scene (not in git)
├── Avatars/                 # Avatar models (not in git)
│   ├── avatar1.fbx
│   ├── avatar2.fbx
│   └── ...
├── Scenes/                  # Scene models (not in git)
│   ├── scene1.blend
│   ├── scene2.blend
│   └── ...
└── Old_3DAssets/           # Legacy assets (not in git)
    └── Scenes/
        └── ...
```

## File Formats Supported

- Blender files: `.blend`, `.blend1`
- 3D models: `.fbx`, `.obj`, `.glb`, `.gltf`

## Notes

- These assets are excluded from Git via `.gitignore` due to their large size
- Total asset size can exceed several GB
- Ensure you have sufficient disk space before downloading
- For questions about asset access, contact the project maintainers

## Verification

To verify your setup is correct, check that the paths in `cfg/paths.yaml` resolve correctly:

```bash
# Check if assets exist
ls -lh Empty.blend Horizon.blend
ls -d Avatars/ Scenes/ Old_3DAssets/
```

If all paths exist, you're ready to use SDAL!

