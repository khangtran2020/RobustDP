#!/bin/bash -l

#SBATCH -J RobustDP  #job name
#SBATCH -o results/logs/mnist-clean.out
#SBATCH -p gpu-all      #queue used
#SBATCH --gres gpu:1    #number of gpus needed, default is 0
#SBATCH -c 1            #number of CPUs needed, default is 1
#SBATCH --mem=16G    #amount of memory needed, default is 4096 MB per core

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=tkhang@hbku.edu.qa

module load cuda11.3/toolkit/11.3.0
conda activate rdp


python main.py  --proj_name "mnist-clean" \
        --gen_mode clean \
        --debug 1 \
        --data mnist \
        --lr 0.01 \
        --epochs 100 \
        --att_mode pgd-clean \
        --device gpu \
        --decay 0.0
