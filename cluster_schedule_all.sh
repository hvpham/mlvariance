#!/bin/bash
source /anaconda/etc/profile.d/conda.sh

conda activate azureml_py36_pytorch

python cluster_schedulel_all.py ~/teamdrive/mlvariance

conda deactivate
