#!/usr/bin/env bash
#SBATCH --job-name Autofish
#SBATCH --partition batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kgrant19@student.aau.dk
#SBATCH --job-name Autofish
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
srun python train_k_fold_val.py