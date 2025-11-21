#!/bin/bash -l
#SBATCH --partition=kamiak,camas,rajagopalan
#SBATCH --requeue
#SBATCH --job-name=till_p0p50p100
#SBATCH --time=1-00:00:00
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --array=1-50%20
#SBATCH --output=/home/a.norouzikandelati/Projects/tillage_mapping/code/kamiak_job_files/out_err_till_p0p50p100/output_split_num_%A_%a.out
#SBATCH --error=/home/a.norouzikandelati/Projects/tillage_mapping/code/kamiak_job_files/out_err_till_p0p50p100/error_split_num_%A_%a.err

set -euo pipefail

echo
echo "--- We are now in $PWD ..."
echo "Job ID: ${SLURM_JOB_ID}  Array ID: ${SLURM_ARRAY_JOB_ID}  Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Time: $(date)"
echo

# Use the env's Python directly (no module/conda activation needed)
PY="$HOME/.conda/envs/gee/bin/python"

# Make sure threaded libs respect cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Ensure no system site-packages sneak in
unset PYTHONPATH

# Split index passed to your script
SPLIT_NUM="${SLURM_ARRAY_TASK_ID}"

# Sanity check inside the srun context (prints once per task)
srun --export=ALL,PYTHONPATH= "$PY" -c "import sys, sklearn; print('exe:', sys.executable); print('sklearn:', sklearn.__version__); print('sklearn file:', sklearn.__file__)"

# Run your job with the same clean environment
srun --export=ALL,PYTHONPATH= "$PY" /home/a.norouzikandelati/Projects/tillage_mapping/code/train_till_models_p0p50p100.py "$SPLIT_NUM"

echo "----- DONE (task ${SLURM_ARRAY_TASK_ID}) -----"