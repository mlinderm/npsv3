#!/usr/bin/env bash

# SLURM template for serial jobs

# Set SLURM options
#SBATCH --mem=48G                 # Job memory request
#SBATCH --cpus-per-task=4  # Number of CPU cores for this job
#SBATCH --partition=gpu-standard        # Partition (queue) 
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --gres=gpu:1

# Print SLURM environment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 

# Start of job info
echo "Starting: "`date +"%D %T"` 

# Your calculations here 
python -u 7.9Initial.py # Add -u for unbuffered output to see real-time progress

# End of job info 
echo "Ending: "`date +"%D %T"`