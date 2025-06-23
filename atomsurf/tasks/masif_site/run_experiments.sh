#!/bin/bash

# Array of batch sizes to test
BATCH_SIZES=(1 8)

# Array of learning rates to test
LEARNING_RATES=(0.001)

# Base experiment name
BASE_NAME="batch_size_experiment_150epochs"

# Function to run a single experiment
run_experiment() {
    local batch_size=$1
    local lr=$2
    local run_name="${BASE_NAME}_bs${batch_size}"
    local comment="Testing batch size ${batch_size} on clean data for 150 epochs"
    
    echo "Running experiment: ${run_name}"
    echo "Comment: ${comment}"
    echo "----------------------------------------"
    
    python atomsurf/tasks/masif_site/train_hydra.py \
        run_name=${run_name} \
        comment="${comment}" \
        loader.batch_size=${batch_size} \
        optimizer.lr=${lr} \
        epochs=150
    
    echo "----------------------------------------"
    echo "Experiment ${run_name} completed"
    echo ""
}

# Main loop to run all experiments
for bs in "${BATCH_SIZES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        run_experiment $bs $lr
        
        # Optional: Add a small delay between experiments
        sleep 2
    done
done

echo "All experiments completed!"
echo "Results are saved in experiment_tracker.csv" 