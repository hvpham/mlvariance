#!/bin/bash
source /anaconda/etc/profile.d/conda.sh

conda activate azureml_py36_pytorch

python cluster_schedule_all.py ~/teamdrive/mlvariance train

conda deactivate
