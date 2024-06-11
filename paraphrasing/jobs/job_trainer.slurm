#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=50:00:00
#SBATCH --job-name="uio-mistral"
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --output=Logs/uiomistral-fine-tune.txt
#SBATCH --error=Logs/uioerror-mistral.txt
#SBATCH --mail-user=annajbj@ntnu.no
#SBATCH --mail-type=ALL
 
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.8.0

CONFIG_PATH="configs/trainer_args.json"

python MT/TrafficAnnouncementsAutomation/paraphrasing/trainer.py $CONFIG_PATH