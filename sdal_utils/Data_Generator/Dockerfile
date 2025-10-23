# Use the official CUDA image as a base
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    xz-utils \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    xvfb \
    libglu1 \
    libxi6 \
    libgconf-2-4 \
    libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

# Install Blender 3.6.9
RUN wget https://download.blender.org/release/Blender3.6/blender-3.6.9-linux-x64.tar.xz -O /tmp/blender.tar.xz && \
    tar -xf /tmp/blender.tar.xz -C /opt && \
    rm /tmp/blender.tar.xz

ENV PATH="/opt/blender-3.6.9-linux-x64:${PATH}"

# Set up a working directory
WORKDIR /workspace

# Install dependencies of adaptive data generator (PyYAML) using pip for Blender's Python
RUN /opt/blender-3.6.9-linux-x64/3.6/python/bin/python3.10 -m ensurepip && \
    /opt/blender-3.6.9-linux-x64/3.6/python/bin/python3.10 -m pip install --upgrade pip && \
    /opt/blender-3.6.9-linux-x64/3.6/python/bin/pip install PyYAML

# Copy the required directories and files
# COPY Avatars /workspace/Avatars
# COPY Scenes /workspace/Scenes
# COPY Dataset /workspace/Dataset
COPY 231109_Adaptive_Data_Generator.py /workspace/231109_Adaptive_Data_Generator.py
COPY Empty.blend /workspace/Empty.blend
COPY Horizon.blend /workspace/Horizon.blend

# Ensure the script has execute permissions
RUN chmod +x /workspace/231109_Adaptive_Data_Generator.py

# Run the script
CMD ["blender", "/workspace/Empty.blend", "--background", "--python", "/workspace/231109_Adaptive_Data_Generator.py"]
