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

# Read num_containers from config.yaml
CONFIG_YAML="./sdal_utils/Data_Generator/config.yaml"
NUM_CONTAINERS=3  # Default fallback
if [ -f "$CONFIG_YAML" ]; then
    # Extract num_containers using grep and sed (simple YAML parsing)
    NUM_CONTAINERS=$(grep -E "^num_containers:" "$CONFIG_YAML" | sed 's/.*num_containers:[[:space:]]*\([0-9]*\).*/\1/' || echo "3")
    # If extraction failed, use default
    if ! [[ "$NUM_CONTAINERS" =~ ^[0-9]+$ ]]; then
        NUM_CONTAINERS=3
    fi
fi

echo "Using $NUM_CONTAINERS parallel containers for data generation"

# Array to store container IDs
declare -a container_ids

# Run Docker containers in parallel with different volume mount points
for i in $(seq 1 $NUM_CONTAINERS); do
    container_name="adaptive_blendcon_$i"
    
    # Check if Dataset_$i directory exists, if not create it
    dataset_dir="./sdal_utils/Data_Generator/Dataset_$i"
    if [ ! -d "$dataset_dir" ]; then
        echo "Creating directory $dataset_dir..."
        mkdir -p "$dataset_dir"
    fi
    # Ensure directory is owned by current user (important for Docker --user flag)
    chmod 755 "$dataset_dir" 2>/dev/null || true
    
    # Check if logs_$i directory exists, if not create it
    logs_dir="./sdal_utils/Data_Generator/logs_$i"
    if [ ! -d "$logs_dir" ]; then
        echo "Creating directory $logs_dir..."
        mkdir -p "$logs_dir"
    fi
    # Ensure directory is owned by current user
    chmod 755 "$logs_dir" 2>/dev/null || true
    
    echo "Running Docker container $container_name..."
    # Run without --rm so we can check exit codes later
    container_id=$(docker run --gpus all -d \
        --user "$(id -u):$(id -g)" \
        -v "$(pwd)/sdal_utils/Data_Generator/Dataset_$i:/workspace/Dataset" \
        -v "$(pwd)/sdal_utils/Data_Generator/logs_$i:/workspace/logs" \
        -v "$(pwd)/sdal_utils/Data_Generator/Avatars:/workspace/Avatars" \
        -v "$(pwd)/sdal_utils/Data_Generator/Scenes:/workspace/Scenes" \
        -v "$(pwd)/sdal_utils/Data_Generator/config.yaml:/workspace/config.yaml" \
        --name "$container_name" adaptive_blendcon)
    container_ids+=("$container_id")
done

# Wait for all containers to finish
while true; do
    all_done=true
    status_line=""
    
    for i in $(seq 1 $NUM_CONTAINERS); do
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

# Check exit codes and report failures
any_failed=false
for i in $(seq 1 $NUM_CONTAINERS); do
    container_name="adaptive_blendcon_$i"
    exit_code=$(docker inspect "$container_name" --format='{{.State.ExitCode}}' 2>/dev/null)
    if [ "$exit_code" != "0" ]; then
        echo "WARNING: Container $container_name failed with exit code $exit_code"
        any_failed=true
    fi
    # Clean up the container
    docker rm "$container_name" 2>/dev/null
done

if [ "$any_failed" = true ]; then
    echo "WARNING: Some containers failed. Check logs for details."
fi

# Always exit 0 - let Python code handle missing data gracefully
exit 0
