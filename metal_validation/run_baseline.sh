#!/bin/bash
#SBATCH --job-name=vina_baseline
#SBATCH --partition=intel
#SBATCH --nodelist=KENTO-GPU8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/cycheng/Vina-GPU-2.1/metal_validation/baseline_run.log
#SBATCH --error=/home/cycheng/Vina-GPU-2.1/metal_validation/baseline_run.err

cd /home/cycheng/Vina-GPU-2.1/metal_validation
echo "=== Job started at $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
python3 run_baseline.py
echo "=== Job finished at $(date) ==="
