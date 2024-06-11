#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=40:00:00
#SBATCH --job-name="mistral-eval"
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100
#SBATCH --mem=256G
#SBATCH --output=Logs/eval_mistral.txt
#SBATCH --error=Logs/error_eval_mistral.txt
#SBATCH --mail-user=annajbj@ntnu.no
#SBATCH --mail-type=ALL
 
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.8.0

CONFIG_PATH="configs/evaluation_args.json"


python MT/TrafficAnnouncementsAutomation/paraphrasing/evaluation.py --config $CONFIG_PATH