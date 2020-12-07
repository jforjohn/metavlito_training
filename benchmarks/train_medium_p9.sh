#!/bin/bash
#SBATCH --job-name="train_medium"
#SBATCH -D /gpfs/projects/bsc28/bsc28301/MetAvlito/training
#SBATCH --output=logs/train_medium_%J.out
#SBATCH --error=logs/train_medium_%J.err
#SBATCH --qos=bsc_cs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:00:00

module purge
module load openmpi gcc cuda cudnn atlas/3.10.3 scalapack/2.0.2 fftw/3.3.7 szip/2.1.1 ffmpeg opencv/3.4.1 python/3.6.5_ML
export PYTHONPATH=.:$PYTHONPATH
date
python benchmarks/train_medium.py
date
