#!/bin/bash
#SBATCH --account=katritch_502
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00

module purge
singularity exec --bind /scratch1/aoxu convert2sdf.icm