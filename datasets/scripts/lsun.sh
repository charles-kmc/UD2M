#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D "/users/cmk2000/cmk2000/Deep learning models/LLMs/Regularised_CDM/datasets"
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o job-%j.output
#SBATCH -e job-%j.error
## Job name
#SBATCH -J LSUN
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=50:30:00
## Memory limit (in megabytes)
#SBATCH --mem=100024
## Specify partition
#SBATCH -p nodes

################# Part-2 Shell script ####################
#===============================
#  Activate Flight Environment
#-------------------------------
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh

# Activate the specific Conda environment
conda activate dmog_env39

if [ $? -eq 0 ]; then
    echo "Conda is active and functioning."
else
    echo "Conda did not initialize properly."
    exit 1
fi
#===========================
#  Create results directory
#---------------------------
RESULTS_DIR="$(pwd)/${SLURM_JOB_NAME}-outputs/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"


#===============================
#  Application launch commands
#-------------------------------
echo "Executing Python script, current working directory is $(pwd)"

# REPLACE THE FOLLOWING WITH YOUR PYTHON SCRIPT
# Assuming the script is located in /users/cmk2000 and named script.py
python3 download_lsun.py -c bedroom -s "train" -o /users/cmk2000/sharedscratch/Datasets/LSUN > "$RESULTS_DIR/lsun_output.txt" 2> "$RESULTS_DIR/lsun_error.txt"


echo "Python script execution complete. Check the output in: $RESULTS_DIR/lsun_output.txt"

