#!/bin/bash

echo "Checking for existing adaptive_blendcon containers..."

# Remove all running containers and images if any exist
containers=$(docker ps -a -q --filter "ancestor=adaptive_blendcon")
if [ -n "$containers" ]; then
    echo "Removing containers..."
    docker rm -f $containers
fi

images=$(docker images -q adaptive_blendcon)
if [ -n "$images" ]; then
    echo "Removing image..."
    docker rmi -f $images
fi

echo "Removing dangling images..."
docker image prune -f

echo "Building adaptive_blendcon Docker image..."
docker build -t adaptive_blendcon "./sdal_utils/Data_Generator"

# Run three Docker containers in parallel with different volume mount points
for i in {1..3}; do
    container_name="adaptive_blendcon_$i"
    
    # Check if Dataset_$i directory exists, if not create it
    dataset_dir="./sdal_utils/Data_Generator/Dataset_$i"
    if [ ! -d "$dataset_dir" ]; then
        echo "Creating directory $dataset_dir..."
        mkdir -p "$dataset_dir"
    fi
    
    # Check if logs_$i directory exists, if not create it
    logs_dir="./sdal_utils/Data_Generator/logs_$i"
    if [ ! -d "$logs_dir" ]; then
        echo "Creating directory $logs_dir..."
        mkdir -p "$logs_dir"
    fi
    
    echo "Running Docker container $container_name..."
    docker run --gpus all -d --rm \
        -v "$(pwd)/sdal_utils/Data_Generator/Dataset_$i:/workspace/Dataset" \
        -v "$(pwd)/sdal_utils/Data_Generator/logs_$i:/workspace/logs" \
        -v "$(pwd)/sdal_utils/Data_Generator/Avatars:/workspace/Avatars" \
        -v "$(pwd)/sdal_utils/Data_Generator/Scenes:/workspace/Scenes" \
        -v "$(pwd)/sdal_utils/Data_Generator/config.yaml:/workspace/config.yaml" \
        --name "$container_name" adaptive_blendcon
done

# Wait for all containers to finish
while true; do
    all_done=true
    status_line=""
    
    for i in {1..3}; do
        container_name="adaptive_blendcon_$i"
        if docker ps -q --filter "name=$container_name" | grep -q .; then
            status_line="$status_line$container_name is still running. "
            all_done=false
        else
            status_line="$status_line$container_name has finished. "
        fi
    done
    
    # Print status on the same line using carriage return
    echo -ne "Checking status of containers... $status_line\r"
    
    if [ "$all_done" = true ]; then
        break
    fi
    
    sleep 10
done

echo -e "\nAll Docker containers have completed their work."
