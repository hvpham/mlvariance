#!/bin/bash
source /anaconda/etc/profile.d/conda.sh

conda activate azureml_py38_PT_and_TF

python cluster_schedule_all.py ~/teamdrive/mlvariance train

conda deactivate
