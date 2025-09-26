import os 

for run in ['jed0033']:
    os.system(f"sbatch scripts/process/regrid/submit_regrid.sh {run}")