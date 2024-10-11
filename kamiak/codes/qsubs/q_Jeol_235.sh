#!/bin/bash
#SBATCH --partition=cahnrs,cahnrs_bigmem,camas,kamiak,rajagopalan,stockle
#SBATCH --requeue
#SBATCH --job-name=235    # Job Name
#SBATCH --time=00-15:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=8GB
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000

###SBATCH -k o
#SBATCH --output=/home/a.norouzikandelati/Projects/Tillage_mapping/codes/error/error_235.o
#SBATCH  --error=/home/a.norouzikandelati/Projects/Tillage_mapping/codes/error/error_235.e
echo
echo "--- We are now in $PWD ..."
echo

## echo "I am Slurm job ${SLURM_JOB_ID}, array job ${SLURM_ARRAY_JOB_ID}, and array task ${SLURM_ARRAY_TASK_ID}."


# module load gcc/7.3.0
module load anaconda3
source /opt/apps/anaconda3/22.10.0/etc/profile.d/conda.sh
source activate gee # conda activate gee # conda when interctive

# ----------------------------------------------------------------
# Gathering useful information
# ----------------------------------------------------------------
echo "--------- environment ---------"
env | grep PBS

echo "--------- where am i  ---------"
echo WORKDIR: ${PBS_O_WORKDIR}
echo HOMEDIR: ${PBS_O_HOME}

echo Running time on host `hostname`
echo Time is `date`
echo Directory is `pwd`

echo "--------- continue on ---------"

# ----------------------------------------------------------------
# Run python code for matrix
# ----------------------------------------------------------------

python /home/a.norouzikandelati/Projects/Tillage_mapping/codes/field_level_featureExtraction_eastWA_data.py 1000 235

echo
echo "----- DONE -----"
echo

exit 0
