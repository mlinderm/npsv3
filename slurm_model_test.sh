#!/usr/bin/env bash

# SLURM template for serial jobs

# Set SLURM options
#SBATCH --job-name=model_benchmark      # Job name
#SBATCH --output=ada_output/%j/slurm_model_test-%j.out # Output file incorporating job ID
#SBATCH --partition=gpu-standard        # Partition (queue) 
#SBATCH --time=00:90:00             # Time limit hrs:min:sec
#SBATCH --mem=8gb                 # Job memory request
#SBATCH --gres=gpu:1      # Requests a GPU

# Print SLURM environment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 

# Start of job info
echo "Starting: "`date +"%D %T"` 

mkdir ada_output/${SLURM_JOB_ID}/images

# python3 /home/cachang/npsv3/img_view.py
# python3 /home/cachang/npsv3/model_test.py "$SLURM_JOB_ID"
python3 /home/cachang/npsv3/dataset_checker.py "$SLURM_JOB_ID"

# End of job info 
echo "Ending: "`date +"%D %T"`