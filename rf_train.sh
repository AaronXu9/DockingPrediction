#!/bin/bash
#SBATCH --account=katritch_502
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00

module purge
module load nvhpc/22.11
module usc
module load conda
mamba activate molpal
python main.py
