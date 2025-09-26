#!/bin/bash
#SBATCH --job-name=coarses # Specify job name
#SBATCH --output=coarses.o%j # name for standard output log file
#SBATCH --error=coarses.e%j # name for standard error output log
#SBATCH --partition=compute
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=0
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/aqua_processing/"

# execute python script in respective environment 
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/aqua_processing/scripts/coarsen_timestep/coarsen.py $1 $2