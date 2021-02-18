from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Datastore
from azureml.core.dataset import Dataset
from azureml.core.compute import ComputeTarget
from azureml.contrib.core.compute.k8scompute import AksCompute
from azureml.contrib.core.k8srunconfig import K8sComputeConfiguration

import os
import shutil
import tempfile
import argparse
import math

def write_run_configs(runs, path):
    f = open(path, "w")
    for run in runs:
        f.write("%s\n%s\n%s\n" % run)
    f.close()


def schedule_train_job(data_folder, result_folder, experiment_name, runs):

    #Create project folder
    project_folder = './cluster_source_folder'
    os.makedirs(project_folder, exist_ok=True)
    shutil.copy('cluster_single_job.py', project_folder)
    shutil.copy('cluster_single_run.sh', project_folder)
    shutil.copy('cluster_single_cifar100.py', project_folder)
    #shutil.copy('cluster_single_test.py', project_folder)
    shutil.copy('resnet.py', project_folder)
    #shutil.copy('utils.py', project_folder)
    write_run_configs(runs, os.path.join(project_folder, 'runs'))


    #Create AML Workspace
    ws = Workspace.from_config()
    #print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

    #Setup experiment
    experiment = Experiment(workspace = ws, name = experiment_name)

    #Set compute target
    #compute_target = ComputeTarget(workspace=ws, name="itplabrl1cl1") #MSR-Lab-RL1
    #compute_target = ComputeTarget(workspace=ws, name="itpeastusv100cl") #MSR-Azure-EastUS-V100-16GB
    #compute_target = ComputeTarget(workspace=ws, name="itpseasiav100cl") #MSR-Azure-SouthEastAsia-V100-16GB
    compute_target = ComputeTarget(workspace=ws, name="itplabrr1cl1") #MSR-Lab-RR1-V100-32GB


    #Create training environment
    #myenv = Environment("training-env")
    
    #myenv.docker.enabled = True
    #myenv.docker.base_image = "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04"
    #myenv.python.conda_dependencies.add_pip_package(pip_package = 'torch')
    #myenv.python.conda_dependencies.add_pip_package(pip_package = 'pickle')

    #curated_env_name = 'AzureML-PyTorch-1.4-GPU'
    #myenv = Environment.get(workspace=ws, name=curated_env_name)
    #myenv.save_to_directory(path=curated_env_name)

    myenv = Environment.from_conda_specification(name='pytorch-1.4-gpu', file_path='./conda_dependencies.yml')

    # Specify a GPU base image
    myenv.docker.enabled = True
    myenv.docker.base_image = "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04"
    
    #Mount teamdrive as blob datastore
    #blob_datastore_name='mlvariance'
    #account_name = "mlvariance"
    #container_name = "teamdrive"
    #sas_token = "https://mlvariance.blob.core.windows.net/teamdrive?sp=racwdl&st=2021-02-12T21:48:33Z&se=2021-03-30T21:48:00Z&sv=2020-02-10&sr=c&sig=byqqcw6NnmDUFAy%2F7PSM%2BuojMdVsWu4vyZjGDJ6ysII%3D"
    #sas_token = "?sp=racwdl&st=2021-02-12T21:48:33Z&se=2021-03-30T21:48:00Z&sv=2020-02-10&sr=c&sig=byqqcw6NnmDUFAy%2F7PSM%2BuojMdVsWu4vyZjGDJ6ysII%3D"
    #account_key = "DN2ow8i0DAgeQCqSolDicmNwD+lbDN+77MkkaiGv6DjbjhJkzZUX1UTV5qo1ZahJQkwUTe18oeyERgfcJ5swOA=="
    
    #blob_datastore = Datastore.register_azure_blob_container(workspace=ws, 
    #                                                        datastore_name=blob_datastore_name, 
    #                                                        account_name=account_name,
    #                                                        container_name=container_name,
    #                                                        account_key=account_key)

    #blob_datastore = datastore = Datastore.get(ws, datastore_name='ml_variance_sas')
    blob_datastore = datastore = Datastore.get(ws, datastore_name='ml_variance_key')

    data_ref = blob_datastore.path("data").as_mount()
    #data_ref.path_on_compute = '/tmp/data'
    
    result_ref = blob_datastore.path("result").as_mount()
    #result_ref.path_on_compute = '/tmp/result'
    
        
    arguments = [str(data_ref),
        str(result_ref),
        ]

    #script = 'cluster_single_test.py'
    script = 'cluster_single_cifar100.py'

    src = ScriptRunConfig(source_directory=project_folder, 
                            script=script,
                            arguments = arguments,
                            compute_target = compute_target,
                            environment = myenv)

    src.run_config.data_references[data_ref.data_reference_name] = data_ref.to_config()
    src.run_config.data_references[result_ref.data_reference_name] = result_ref.to_config()

    k8sconfig = K8sComputeConfiguration()
    k8s = dict()
    k8s['gpu_count'] = 1
    k8s['preemption_allowed'] = True
    #k8s['preemption_allowed'] = False
    k8sconfig.configuration = k8s
    src.run_config.cmk8scompute = k8sconfig

    experiment.submit(config=src)

def list_cifar100_holdout_runs(local_result_folder, no_runs):
    mode = 'holdout'
    holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushrooms', 'porcupine', 'ray']

    runs_list = []
    for holdout_class in holdout_classes:
        for ratio in [0,3,6,9]:
            for run_id in range(no_runs):
                outputs_file = os.path.join(local_result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, ratio), 'outputs_%s' % run_id)
                if not os.path.isfile(outputs_file):
                    script = "cluster_single_cifar100.py"
                    args = "%s %s %d %d" % (mode, holdout_class, ratio, run_id)
                    log = "cifar100_%s_%s_%d_%d" % (mode, holdout_class, ratio, run_id)
                    runs_list.append((script, args, log))
        
    return runs_list

parser = argparse.ArgumentParser(description='Schedule experiment runs on the clusters with holdout CIFAR-100 and Resnet18')
parser.add_argument('local_teamdrive_folder', help='local teamdrive folder')

args = parser.parse_args()

local_teamdrive_folder = args.local_teamdrive_folder

NO_RUNS = 100

runs_list = list_cifar100_runs(os.path.join(local_teamdrive_folder, 'result'), NO_RUNS, mode = 'holdout')

MAX_NO_JOBS = 64

#NO_RUNS_PER_JOB = 25
NO_RUNS_PER_JOB = math.ceil(len(runs_list)/MAX_NO_JOBS)
no_jobs = math.ceil(len(runs_list)/NO_RUNS_PER_JOB)
for job_id in range(no_jobs):
    start = job_id * NO_RUNS_PER_JOB
    end = min((job_id + 1) * NO_RUNS_PER_JOB, len(runs_list))
    runs = runs_list[start:end]

    schedule_train_job("data", "result", 'modelvar-train-rn18-holdout', runs)



