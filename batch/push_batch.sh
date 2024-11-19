#!/bin/bash
#SBATCH --time=3-00:00:00                  # Job run time (hh:mm:ss)
#SBATCH --account=myuasa2           # Replace "account_name" with an account available to you
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=24              # Number of task (cores/ppn) per node
#SBATCH --mem-per-cpu=3375               # Memory per core (value in MBs)

#
cd ${SLURM_SUBMIT_DIR}

# Run the serial executable