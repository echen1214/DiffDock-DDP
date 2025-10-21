#!/bin/bash

#SBATCH --cpus-per-task 12
#SBATCH --time=3:00:00
##SBATCH --mail-type=END
##SBATCH --mail-user=eac709@nyu.edu
#SBATCH --mem=30G
#SBATCH --job-name=cache-hetgraphs
#SBATCH --output=250612_diffdock_tune/logs/slurm-%A.out

singularity exec --nv --overlay /scratch/jeg10045/zhang_lab/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/ubuntu-24.04.sif ./gen-diffdock-cache.sh
