#!/bin/bash
#SBATCH --account=katritch_502
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00

module purge
eval "$(conda shell.bash hook)"
# module load conda
conda activate molpal
python predict.py ./predict.sdf
