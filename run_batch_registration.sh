#!/bin/bash

# Ensure the script is run by the Bash interpreter
# set -e: The script will exit immediately if any command fails.
# set -u: The script will error and exit if an undefined variable is used.
# set -o pipefail: The exit status of a pipeline is the exit status of the last command to fail.
set -euo pipefail

# --- 1. Define directory paths ---
BASE_DIR="."
BOUNDARY_DIR="${BASE_DIR}/boundaries/mouse1_coronal"
SPOTS_DIR="${BASE_DIR}/decoded_spots/mouse1_coronal"
OUTPUT_DIR="${BASE_DIR}/normalized_results/mouse1_coronal_parallel"

# --- 2. Define parallel processing parameters ---
# Use nproc to get the number of cores, -1 is to leave one core for the system to prevent freezing
NUM_CORES=$(($(nproc) - 1))
echo "Will use ${NUM_CORES} cores for parallel computation."

# --- 3. Check and create the output directory ---
echo "Checking output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
echo "Output will be saved to: ${OUTPUT_DIR}"
echo "----------------------------------------------------"

# --- 4. Define and export the processing function ---
# Define and export the function before the main logic
# This is key to solving the "not a function" error
process_file() {
    local BOUNDARY_FILE="$1"
    
    # Extract the core filename (identifier) from the boundary file
    local FILENAME
    FILENAME=$(basename "${BOUNDARY_FILE}")
    local IDENTIFIER
    IDENTIFIER="${FILENAME%.csv}"

    # Get directory paths from environment variables
    # This is more robust than relying on external variables
    local SPOTS_DIR_LOCAL="$SPOTS_DIR"
    local OUTPUT_DIR_LOCAL="$OUTPUT_DIR"

    # Construct the paths for the corresponding spots file and output file
    local SPOTS_FILE="${SPOTS_DIR_LOCAL}/spots_${IDENTIFIER}.csv"
    local OUTPUT_FILE="${OUTPUT_DIR_LOCAL}/normalized_${IDENTIFIER}.csv"
    
    # Check if the corresponding spots file exists
    if [ ! -f "${SPOTS_FILE}" ]; then
        # In a parallel environment, use stderr to output warning messages
        echo "Warning: ${SPOTS_FILE} not found, skipping ${IDENTIFIER}." >&2
    else
        # This echo will be captured by parallel; can be viewed in real-time with --line-buffer
        echo "Starting processing: ${IDENTIFIER}"
        
        # Activate conda environment (if needed, this is good practice)
        # If your python and libraries are in the base environment, you can comment out the next two lines
        # source /path/to/conda/etc/profile.d/conda.sh
        # conda activate merfish_env
        
        # Execute the Python script
        # Redirecting output keeps the progress bar clean, but can be removed for debugging
        python merfish_registration_cli.py \
            --spots "${SPOTS_FILE}" \
            --boundaries "${BOUNDARY_FILE}" \
            --output "${OUTPUT_FILE}" >/dev/null 2>&1
        
        echo "Finished processing: ${IDENTIFIER}"
    fi
}
# Export the function and necessary variables to subshells
export -f process_file
export SPOTS_DIR OUTPUT_DIR

# --- 5. Use GNU Parallel to process files in parallel ---
echo "Starting parallel processing..."

# Use find to locate all boundary files, separated by null characters
# --bar: Display a progress bar
# --eta: Display the estimated time of arrival (ETA)
# --joblog: Create a job log
# --line-buffer: Output the echo from each job in real-time instead of waiting for the job to finish
find "${BOUNDARY_DIR}" -maxdepth 1 -name "*.csv" -print0 | \
    parallel -0 --bar --eta -j "${NUM_CORES}" --joblog "${OUTPUT_DIR}/parallel_joblog.log" --line-buffer process_file {}

echo "----------------------------------------------------"
echo "All files have been processed in parallel!"
echo "Job log has been saved to ${OUTPUT_DIR}/parallel_joblog.log"