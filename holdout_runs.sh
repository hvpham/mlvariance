#!/bin/bash
now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE="logs/artificial_runs_$now.log"

source /anaconda/etc/profile.d/conda.sh

exec &>$LOG_FILE

conda activate azureml_py36_pytorch

python main_artificial.py

conda deactivate
