#!/bin/bash
#
#SBATCH --job-name=ham_classifier
#SBATCH --output=/mnt/beegfs/work/tiblias/arxiv2024-ham-classifier/logs/slurm_logs.txt
#SBATCH --mail-user=federico.tiblias@tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -q yolo
#SBATCH -p yolo
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1


cd /ukp-storage-1/tiblias/arxiv2024-ham-classifier/
source .env
export WANDB_API_KEY=$WANDB_API_KEY
export PATH=$PATH:/ukp-storage-1/tiblias/miniconda/envs/qse-gpu/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/tiblias/miniconda/envs/qse-gpu/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/tiblias/miniconda/lib

source /ukp-storage-1/tiblias/miniconda/bin/activate qse-gpu

python -m ham_classifier --arch ham_sim --dataset sst2 --mode sweep
