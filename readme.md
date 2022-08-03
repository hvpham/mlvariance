# Code for the Model Variability project

This is the code repo for the research project Model Variablilty to investigate the variability of deep learning models and how variance based metrics can be used to debug deep learning models.

This is a summary of how to runs various scripts. The process of running the experiment is as following:

1. Prepare the data
2. Train the models
3. Analyze and generate result

# Env preparation:
1. Use the "azureml_py36_pytorch" anaconda env as the base (Already in the new node).
2. You also need to mount the mlvariance to "~/teamdrive/mlvariance".
3. Install packages to "azureml_py36_pytorch" using:
- **python3 -m pip install --disable-pip-version-check --extra-index-url https://azuremlsdktestpypi.azureedge.net/K8s-Compute/D58E86006C65 azureml_contrib_k8s**

# Data preparation:
Data preparation scripts are located under the folder **prepare_data**.

## Prepare COMPAS dataset
**_prepare_artificial_compas.py_** and **_prepare_holdout_compas.py_** prepare COMPAS data for two scenarios: artificial correlated features and holding out clustered sets.

The path to the data is hard coded in the scripts. Change them to point to the correct path if needed.

## Prepare CIFAR10 dataset
**_prepare_holdout_cifar10.py_** prepares CIFAR10 data for holdout scenarios.

The script takes two arguments:

1. The **_data_folder_** that would contain the prepared data. This should be pointed to *~/teamdrive/mlvariance/data* if run from GPU dev machine.

2. The **_mode_** which represents the scenario:

- *holdout*: holdout a portion of class 0

- *holdout-dup*: holdout a portion of class 0 with duplication so the number of training examples is evently distributed across all classes

- *augmentation*: convert a portion of class 0 to grayscale

- *augmentation-all*: convert a portion of all classes to grayscale

## Prepare CIFAR100 dataset
**_prepare_holdout_cifar100-cifar10.py_** prepares CIFAR100 data for holdout scenarios.

The script takes two arguments:

1. The **_data_folder_** that would contain the prepared data. This should be pointed to *~/teamdrive/mlvariance/data* if run from GPU dev machine.

2. The **_mode_** which represents the scenario:

- *holdout*: holdout a portion of a subclass. The list of subclasses are hard coded in the script.

**_local_export_images_cifar100.py_**: export images to display in html report

# Training deep learning models

There are various training scripts for different datasets for COMPAS, CIFAR10, and CIFAR100

## COMPAS training:

To train the models for the COMPAS dataset, you can run one of several scripts
1. **_main_compas.py_**: for training with original COMPAS dataset
1. **_main_artificial_compas.py_**: for training using COMPAS dataset with artificial correlated features
1. **_main_holdout_compas.py_**: for training using COMPAS dataset with holdout clusted set

## CIFAR10 and CIFAR100 training:

To train models for CIFAR10 and CIFAR100, the AML platform would be used. To schedule the training jobs, the script **_cluster_schedulel_all.py_** can be used. This is a master script that queue jobs to the AML cluster.
The script take a single argument **_local_teamdrive_folder_** which should point to *~/teamdrive/mlvariance*.
There should be a **_config.json_** file with the subscription info for the AML cluster. This is the information from [https://dev.azure.com/msresearch/GCR/_wiki/wikis/GCR.wiki/3438/AML-K8s-(aka-ITP)-Overview] under the table Connection Details.

1. To train CIFAR10 models uncomment the line **_runs_list = list_cifar10_runs(...)_**. This function generate the jobs list which consist of single run of **_cluster_single_cifar10.py_**
2. To train CIFAR100 models uncomment the line **_runs_list = list_cifar100_runs(...)_**. This function generate the jobs list which consist of single run of **_cluster_single_cifar100.py_**

# Analyze results
There are several scripts that can be used to generate the analysis result. There are some analysis scripts under the **_analysis_** folder.

## Analyze COMPAS results

There are several script to analyze COMPAS result under the **_analysis_** folder.

1. **_analyze.py**: analyze COMPAS result without holdout
2. **_analyze_artificial.py**: analyze COMPAS result with artificial correlated features 
3. **_analyze_holdout.py**: analyze COMPAS result with holdout cluster set

## Analyze CIFAR10 results

Run the **_analyze_cifar10.py_** script to analyze the accuracy of the CIFAR10 models. This script takes one argument **_result_folder_** which should be pointed to *~/teamdrive/mlvariance/result* if run in GPU Dev machine.

## Analyze CIFAR100 results

There are several analysis script for CIFAR100.

### Some should be run on the AML cluster:

1. **_cluster_single_detail_analysis_cifar100.py_**: runs this via **_cluster_schedulel_all.py_** by uncommenting **_runs_list = list_cifar100_holdout_analysis_runs(..., "cluster_single_detail_analysis_cifar100.py")_**. This generates detail analysis for each sample and also create ranking results.

2. **_cluster_single_saliency_cifar100.py_**: runs this via **_cluster_schedulel_all.py_** by uncommenting **_runs_list = list_cifar100_holdout_generate_map_runs(...)_**. This generates gradcam images for all samples.

3. **_cluster_single_detail_analysis_cifar100_html.py_**: runs this via **_cluster_schedulel_all.py_** by uncommenting **_runs_list = list_cifar100_holdout_analysis_runs(..., "cluster_single_detail_analysis_cifar100_html.py")_**. This generates detail analysis in the form of html pages for each sample which includes gradcam images.

### Some are local scripts:

1. **_analyze_cifar100.py_**: analyze CIFAR100 models accuracy
2. **_local_merge_rank_report.py_**: to merge the result once all **_cluster_single_detail_analysis_cifar100.py_** has finished.



