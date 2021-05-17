#!/bin/bash
now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE="logs/cifar100_holdout_evaluate_$now.log"

source /home/hvpham/anaconda3/etc/profile.d/conda.sh

exec &>$LOG_FILE

conda activate aml_pytorch_python36

python local_evaluate_cifar100.py 25 "/local2/teamdrive/data" "/local2/teamdrive/result" "holdout"

conda deactivate
