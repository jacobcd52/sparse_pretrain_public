#!/bin/bash
# Run CARBS sweep in background with full logging

cd /root/global_circuits

# Create output directory
OUTPUT_DIR="outputs/carbs_runs"
mkdir -p $OUTPUT_DIR

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/carbs_sweep_${TIMESTAMP}.log"

echo "Starting CARBS sweep at $(date)" | tee $LOG_FILE
echo "Log file: $LOG_FILE"
echo "Task: dummy_pronoun"
echo "Iterations: 32, Parallel: 8, Steps: 2000"

# Run the sweep with nohup
# Using 32 iterations with 8 parallel = 256 total runs (as per paper)
nohup python -u scripts/run_carbs_sweep.py \
    --task dummy_pronoun \
    --iterations 32 \
    --parallel 8 \
    --steps 2000 \
    --target-loss 0.15 \
    >> $LOG_FILE 2>&1 &

PID=$!
echo "Started sweep with PID: $PID" | tee -a $LOG_FILE
echo $PID > $OUTPUT_DIR/sweep_pid.txt

echo ""
echo "Sweep is running in background. To monitor:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop:"
echo "  kill $PID"

