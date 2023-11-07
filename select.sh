#!/bin/bash
#SBATCH --account=katritch_502
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00

module purge
eval "$(conda shell.bash hook)"
# module load conda
conda activate molpal
python predict.py "$@"
# ./predict.sdf --output_file=./analysis/top100K_RF_preds.sdf
