#!/bin/bash

# Launch script for masif_ligand training experiments
# This script creates tmux sessions for each training configuration
# and distributes them across available GPU devices (3 per device)

set -e  # Exit on error

# Configuration
SCRIPT_DIR="/root/atomsurf/atomsurf/tasks/masif_ligand"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
BASE_DIR="${SCRIPT_DIR}"
CONDA_ENV="atomsurf"

# Available GPU devices (modify this based on your system)
# Check available devices with: nvidia-smi --list-gpus
DEVICES=(0 2 3)  # Modify this array to match your available GPUs
MAX_JOBS_PER_DEVICE=2

# Variable configurations
MAX_DROP_VALUES=(0 2)
SEEDS=(2024 2025 42)
NORM_TYPES=("batch")

# Base command (fixed parameters) - all on one line for tmux
BASE_CMD="python3 ${TRAIN_SCRIPT} encoder=pronet_gvpencoder optimizer.lr=0.0001 scheduler=reduce_lr_on_plateau epochs=100 loader.batch_size=16 loader.num_workers=16 diffusion_net.use_bn=true diffusion_net.use_layernorm=false diffusion_net.init_time=2.0 diffusion_net.init_std=2.0 train.save_top_k=5 train.early_stoping_patience=500 exclude_failed_patches=True loader.drop_last=True"

# Tracking variables for device assignment
declare -A DEVICE_COUNTS
for device in "${DEVICES[@]}"; do
    DEVICE_COUNTS[$device]=0
done

# Track current device index for round-robin
CURRENT_DEVICE_INDEX=0

# Function to get next available device (modifies global variable SELECTED_DEVICE)
get_next_device() {
    local attempts=0
    local max_attempts=${#DEVICES[@]}
    
    # Try to find a device with available slots
    while [ $attempts -lt $max_attempts ]; do
        # Get device from current index (round-robin)
        SELECTED_DEVICE=${DEVICES[$CURRENT_DEVICE_INDEX]}
        
        # Check if this device has available slots
        if [ ${DEVICE_COUNTS[$SELECTED_DEVICE]} -lt $MAX_JOBS_PER_DEVICE ]; then
            # This device has room, use it
            DEVICE_COUNTS[$SELECTED_DEVICE]=$((DEVICE_COUNTS[$SELECTED_DEVICE] + 1))
            CURRENT_DEVICE_INDEX=$(((CURRENT_DEVICE_INDEX + 1) % ${#DEVICES[@]}))
            return
        fi
        
        # Move to next device
        CURRENT_DEVICE_INDEX=$(((CURRENT_DEVICE_INDEX + 1) % ${#DEVICES[@]}))
        attempts=$((attempts + 1))
    done
    
    # If all devices are full, find the one with lowest count (shouldn't happen with proper distribution)
    local min_jobs=999
    SELECTED_DEVICE=${DEVICES[0]}
    for device in "${DEVICES[@]}"; do
        if [ ${DEVICE_COUNTS[$device]} -lt $min_jobs ]; then
            min_jobs=${DEVICE_COUNTS[$device]}
            SELECTED_DEVICE=$device
        fi
    done
    
    DEVICE_COUNTS[$SELECTED_DEVICE]=$((DEVICE_COUNTS[$SELECTED_DEVICE] + 1))
    CURRENT_DEVICE_INDEX=$(((CURRENT_DEVICE_INDEX + 1) % ${#DEVICES[@]}))
}

# Function to reset device counter when moving to next device (no longer needed with new logic)
reset_device_if_needed() {
    # This function is kept for compatibility but doesn't need to do anything
    # The new round-robin logic handles device distribution automatically
    :
}

# Counter for total experiments
TOTAL_EXPERIMENTS=0
LAUNCHED_EXPERIMENTS=0

# Calculate total number of experiments
for max_drop in "${MAX_DROP_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for norm_type in "${NORM_TYPES[@]}"; do
            TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
        done
    done
done

echo "=========================================="
echo "Launching ${TOTAL_EXPERIMENTS} training experiments"
echo "Devices: ${DEVICES[@]}"
echo "Max jobs per device: ${MAX_JOBS_PER_DEVICE}"
echo "=========================================="
echo ""

# Create all experiments
for max_drop in "${MAX_DROP_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for norm_type in "${NORM_TYPES[@]}"; do
            # Get device for this experiment (function modifies SELECTED_DEVICE directly)
            get_next_device
            device=$SELECTED_DEVICE
            
            # Debug: Show device assignment
            echo "[${LAUNCHED_EXPERIMENTS}/${TOTAL_EXPERIMENTS}] Device assignment: device=${device}, device_counts: $(for d in "${DEVICES[@]}"; do echo -n "${d}=${DEVICE_COUNTS[$d]} "; done)"
            
            # Create run name with placeholders replaced
            run_name="hybrid_gvp_3layers_filtered_seed${seed}_drop${max_drop}_norm${norm_type}"
            
            # Create tmux session name (tmux names have restrictions)
            session_name="masif_ligand_seed${seed}_drop${max_drop}_norm${norm_type}"
            # Replace any characters that might cause issues in tmux session names
            session_name=$(echo "$session_name" | sed 's/[^a-zA-Z0-9_]/_/g')
            
            # Create detached tmux session and run command
            echo "  Launching: seed=${seed}, max_drop=${max_drop}, norm_type=${norm_type}, device=${device}"
            echo "  Session: ${session_name}"
            echo "  Run name: ${run_name}"
            
            # Build and execute command directly (avoid nested quote issues)
            # Create tmux session in detached mode and run command
            # Note: tmux sessions need to source conda setup, so we use bash -c with conda initialization
            tmux new-session -d -s "${session_name}" -c "${BASE_DIR}" \
                bash -c "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${BASE_DIR} && ${BASE_CMD} seed=${seed} loader.max_drop_per_batch=${max_drop} cfg_head.norm_type=${norm_type} device=${device} run_name='${run_name}' 2>&1 | tee ${BASE_DIR}/${session_name}.log"
            
            LAUNCHED_EXPERIMENTS=$((LAUNCHED_EXPERIMENTS + 1))
            
            # Small delay to avoid overwhelming the system
            sleep 2
            
            # Reset device counter if we've filled this device
            reset_device_if_needed $device
        done
    done
done

echo ""
echo "=========================================="
echo "All ${LAUNCHED_EXPERIMENTS} experiments launched!"
echo ""
echo "To monitor sessions:"
echo "  tmux list-sessions | grep masif_ligand"
echo ""
echo "To attach to a session:"
echo "  tmux attach -t <session_name>"
echo ""
echo "To kill a session:"
echo "  tmux kill-session -t <session_name>"
echo ""
echo "To view logs:"
echo "  tail -f <session_name>.log"
echo "=========================================="

