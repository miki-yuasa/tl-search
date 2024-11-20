#!/bin/bash
#SBATCH --time=3-00:00:00                  # Job run time (hh:mm:ss)
#SBATCH --account=myuasa2           # Replace "account_name" with an account available to you
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=24              # Number of task (cores/ppn) per node
#SBATCH --mem-per-cpu=515500               # Memory per core (value in MBs)
#SBATCH --job-name=tl_search_exh_push          # Job name
#SBATCH --output=tl_search_exh_push_%j.out   # Output file name
#SBATCH --error=tl_search_exh_push_%j.err    # Error file name
#SBATCH --gres=gpu:A100:1                      # Number of GPUs
#SBATCH --partition=IllinoisComputes-GPU

# Load the necessary modules
cd /projects/illinois/eng/aerospace/huytran1/myuasa2/git/tl-search
module load anaconda/2023-Mar/3
source /projects/illinois/eng/aerospace/huytran1/myuasa2/.cache/pypoetry/virtualenvs/tl-search-WyzSwqE--py3.10/bin/activate

# Run the serial executable
/projects/illinois/eng/aerospace/huytran1/myuasa2/.cache/pypoetry/virtualenvs/tl-search-WyzSwqE--py3.10/bin/python /projects/illinois/eng/aerospace/huytran1/myuasa2/git/tl-search/scripts/main_tl_exhaustive_push.py