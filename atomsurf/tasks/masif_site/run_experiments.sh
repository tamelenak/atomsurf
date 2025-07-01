#!/bin/bash

# Set this to get detailed error logs if something fails
export HYDRA_FULL_ERROR=1

# Array of encoders to test
ENCODERS=("pronet_gvpencoder" "graph_only_pronet")

# Array of batch sizes to test
BATCH_SIZES=(16 64)

# Number of epochs for each run
EPOCHS=50

# Function to run a single experiment
run_experiment() {
    local encoder=$1
    local batch_size=$2
    local run_name="${encoder}_bs${batch_size}"
    local comment="Encoder: ${encoder}, Batch Size: ${batch_size}, Epochs: ${EPOCHS}"
    
    echo "========================================"
    echo "Running experiment: ${run_name}"
    echo "Comment: ${comment}"
    echo "----------------------------------------"
    
    python atomsurf/tasks/masif_site/train_hydra.py \
        encoder=${encoder} \
        run_name=${run_name} \
        comment="${comment}" \
        loader.batch_size=${batch_size} \
        epochs=${EPOCHS}
    
    echo "----------------------------------------"
    echo "Experiment ${run_name} completed"
    echo "========================================"
    echo ""
}

# Main loop to run all experiments
for encoder in "${ENCODERS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        run_experiment $encoder $bs
        
        # Optional: Add a small delay between experiments
        sleep 5
    done
done

echo "All experiments completed!"
echo "Results are saved in experiment_tracker.csv" 