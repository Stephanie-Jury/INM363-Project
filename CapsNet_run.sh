#! /bin/bash

#SBATCH --job-name='CapsNet'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stephanie.jury@city.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output job%J.out
#SBATCH --error job%J.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:2

module load cuda/10.0

python3 'CapsNet_train_test.py' > CapsNet_train_test.txt
