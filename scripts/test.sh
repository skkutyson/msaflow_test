#!/bin/bash                                                     
#SBATCH --job-name=cuda-test
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX6000:1                                                                                           
#SBATCH --partition=normal
#SBATCH --time=00:05:00                                                                                                
#SBATCH --output=/home/yjlee4/MSA_FLOW/logs/cuda_test_%j.out                                                           
  
apptainer exec --nv \                                                                                                  
    --bind /home/yjlee4/MSA_FLOW:/mnt/MSA_FLOW \                
    /home/yjlee4/msaflow_sandbox \                                                                                     
    bash -c 'cd /mnt/MSA_FLOW && uv run python -c "import torch; print(torch.__version__); print(\"cuda:\", torch.cuda.is_available())"'