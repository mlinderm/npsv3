#!/usr/bin/env bash

# SLURM template for serial jobs

# Set SLURM options
#SBATCH --job-name=model_benchmark      # Job name
#SBATCH --output=ada_output/%A/%a/slurm_model_test-%A-%a.out # Output file incorporating job ID
#SBATCH --array=0-5
#SBATCH --partition=gpu-standard        # Partition (queue) 
#SBATCH --time=00:90:00             # Time limit hrs:min:sec
#SBATCH --mem=8gb                 # Job memory request
#SBATCH --gres=gpu:1      # Requests a GPU

DATASETS=("HG00096" "HG00171" "HG00268" "HG00358" "HG00438" "HG00512")
CURRENT_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Print SLURM environment variables
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 

# Start of job info
echo "Starting: "`date +"%D %T"` 

mkdir ada_output/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}/images

python3 /home/cachang/npsv3/dataset_checker.py ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID} $CURRENT_DATASET
mv ada_output/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID} ada_output/${SLURM_ARRAY_JOB_ID}/${DATASETS[$SLURM_ARRAY_TASK_ID]}

# End of job info 
echo "Ending: "`date +"%D %T"`