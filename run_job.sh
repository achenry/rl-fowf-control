#!/bin/bash
#SBATCH --nodes=20
#SBATCH --gres=gpu:1
#SBATCH --partition=ami100
#SBATCH --time=24:00:00
#SBATCH --job-name=run_rl_wf_rocm
#SBATCH --output=run_rl_wf_rocm.%j.out

module purge
module load anaconda
conda activate rl_wf_env_rocm

cd /projects/aohe7145/projects/rl-fowf-control/
python CleanRL_PSDDPG.py
