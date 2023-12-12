#!/bin/bash
#SBATCH --account=katritch_502
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

module purge
eval "$(conda shell.bash hook)"
# module load conda
conda activate molpal
python predict.py
