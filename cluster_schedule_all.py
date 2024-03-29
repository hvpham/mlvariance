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
        f.write("%s\n%s\n%s\n" % (run[0], run[1], run[2]))
    f.close()


def schedule_train_job(data_folder, result_folder, experiment_name, runs, GPU):

    # Create project folder
    project_folder = './cluster_source_folder'
    os.makedirs(project_folder, exist_ok=True)
    shutil.copy('cluster_single_job.py', project_folder)
    shutil.copy('cluster_single_run.sh', project_folder)
    shutil.copy('cluster_single_cifar100.py', project_folder)
    shutil.copy('cluster_single_val_test_cifar100.py', project_folder)
    shutil.copy('cluster_single_train_more_cifar100.py', project_folder)
    shutil.copy('cluster_single_cifar10.py', project_folder)
    shutil.copy('cluster_single_detail_analysis_cifar100.py', project_folder)
    shutil.copy('cluster_single_detail_analysis_cifar100_html.py', project_folder)
    shutil.copy('cluster_single_saliency_cifar100.py', project_folder)
    shutil.copy('cluster_single_evaluate_cifar100.py', project_folder)
    shutil.copy('sort_script.txt', project_folder)
    #shutil.copy('cluster_single_test.py', project_folder)
    shutil.copy('resnet.py', project_folder)
    #shutil.copy('utils.py', project_folder)

    shutil.copy('cluster_single_amazon.py', project_folder)
    
    model_folder = project_folder + "/models"
    os.makedirs(model_folder, exist_ok=True)
    shutil.copy('models/__init__.py', model_folder)
    
    han_folder = model_folder + "/HAN"
    os.makedirs(han_folder, exist_ok=True)
    shutil.copy('models/HAN/__init__.py', han_folder)
    shutil.copy('models/HAN/han.py', han_folder)
    shutil.copy('models/HAN/sent_encoder.py', han_folder)
    shutil.copy('models/HAN/word_encoder.py', han_folder)

    write_run_configs(runs, os.path.join(project_folder, 'runs'))

    # Create AML Workspace
    ws = Workspace.from_config()
    #print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

    # Setup experiment
    experiment = Experiment(workspace=ws, name=experiment_name)

    # Set compute target
    # compute_target = ComputeTarget(workspace=ws, name="itplabrl1cl1") #MSR-Lab-RL1
    # compute_target = ComputeTarget(workspace=ws, name="itpeastusv100cl") #MSR-Azure-EastUS-V100-16GB
    # compute_target = ComputeTarget(workspace=ws, name="itpseasiav100cl") #MSR-Azure-SouthEastAsia-V100-16GB
    compute_target = ComputeTarget(workspace=ws, name="itplabrr1cl1")  # MSR-Lab-RR1-V100-32GB

    # Create training environment
    #myenv = Environment("training-env")

    #myenv.docker.enabled = True
    #myenv.docker.base_image = "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04"
    #myenv.python.conda_dependencies.add_pip_package(pip_package = 'torch')
    #myenv.python.conda_dependencies.add_pip_package(pip_package = 'pickle')

    #curated_env_name = 'AzureML-PyTorch-1.4-GPU'
    #myenv = Environment.get(workspace=ws, name=curated_env_name)
    # myenv.save_to_directory(path=curated_env_name)

    myenv = Environment.from_conda_specification(name='pytorch-1.4-gpu', file_path='./conda_dependencies.yml')

    # Specify a GPU base image
    myenv.docker.enabled = True
    myenv.docker.base_image = "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04"

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
    #script = 'cluster_single_cifar100.py'
    script = 'cluster_single_job.py'

    src = ScriptRunConfig(source_directory=project_folder,
                          script=script,
                          arguments=arguments,
                          compute_target=compute_target,
                          environment=myenv)

    src.run_config.data_references[data_ref.data_reference_name] = data_ref.to_config()
    src.run_config.data_references[result_ref.data_reference_name] = result_ref.to_config()

    k8sconfig = K8sComputeConfiguration()
    k8s = dict()
    if GPU:
        k8s['gpu_count'] = 1
    else:
        k8s['gpu_count'] = 0
    #k8s['preemption_allowed'] = True
    k8s['preemption_allowed'] = False
    k8sconfig.configuration = k8s
    src.run_config.cmk8scompute = k8sconfig

    experiment.submit(config=src)


def check_done(path):
    if os.path.isfile(path):
        if os.stat(path).st_size > 0:
            return True
    return False


def list_cifar100_holdout_runs(local_result_folder, no_runs):
    mode = 'holdout'
    #holdout_class_list = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
    #holdout_class_list = ['caterpillar', 'mushroom', 'porcupine', 'ray']

    aquatic_mammals_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale']  # 0
    fish_list = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']  # 1
    #fish_list = ['aquarium_fish', 'flatfish', 'shark', 'trout']
    flower_list = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']  # 2
    fruit_and_vegetables_list = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']  # 3
    #fruit_and_vegetables_list = ['apple', 'orange', 'pear', 'sweet_pepper']
    insects_list = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']  # 4
    #insects_list = ['bee', 'beetle', 'butterfly', 'cockroach']
    medium_mammals_list = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']  # 5
    #medium_mammals_list = ['fox', 'possum', 'raccoon', 'skunk']
    people_list = ['baby', 'boy', 'girl', 'man', 'woman']  # 6
    reptiles_list = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']  # 7
    small_mammals_list = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']  # 8
    trees_list = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']  # 9

    holdout_class_list = []
    holdout_class_list.extend(aquatic_mammals_list)  # 0
    holdout_class_list.extend(fish_list)  # 1
    holdout_class_list.extend(flower_list)  # 2
    holdout_class_list.extend(fruit_and_vegetables_list)  # 3
    holdout_class_list.extend(insects_list)  # 4
    holdout_class_list.extend(medium_mammals_list)  # 5
    holdout_class_list.extend(people_list)  # 6
    holdout_class_list.extend(reptiles_list)  # 7
    holdout_class_list.extend(small_mammals_list)  # 8
    holdout_class_list.extend(trees_list)  # 9

    runs_list = []
    for holdout_class in holdout_class_list:
        # for ratio in [0,3,6,9]:
        # for ratio in [1,2,4,5,7,8,10]:
        for ratio in range(11):
            # for ratio in [10]:
            for run_id in range(no_runs):
                outputs_file = os.path.join(local_result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, ratio), 'outputs_%s' % run_id)
                if not check_done(outputs_file):
                    script = "cluster_single_cifar100.py"
                    args = "%s %s %d %d" % (mode, holdout_class, ratio, run_id)
                    log = "cifar100_%s_%s_%d_%d" % (mode, holdout_class, ratio, run_id)
                    runs_list.append((script, args, log, True))

    return runs_list


def list_cifar100_holdout_runs(local_result_folder, no_runs):
    mode = 'holdout'
    #holdout_class_list = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
    #holdout_class_list = ['caterpillar', 'mushroom', 'porcupine', 'ray']

    aquatic_mammals_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale']  # 0
    fish_list = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']  # 1
    #fish_list = ['aquarium_fish', 'flatfish', 'shark', 'trout']
    flower_list = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']  # 2
    fruit_and_vegetables_list = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']  # 3
    #fruit_and_vegetables_list = ['apple', 'orange', 'pear', 'sweet_pepper']
    insects_list = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']  # 4
    #insects_list = ['bee', 'beetle', 'butterfly', 'cockroach']
    medium_mammals_list = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']  # 5
    #medium_mammals_list = ['fox', 'possum', 'raccoon', 'skunk']
    people_list = ['baby', 'boy', 'girl', 'man', 'woman']  # 6
    reptiles_list = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']  # 7
    small_mammals_list = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']  # 8
    trees_list = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']  # 9

    holdout_class_list = []
    holdout_class_list.extend(aquatic_mammals_list)  # 0
    holdout_class_list.extend(fish_list)  # 1
    holdout_class_list.extend(flower_list)  # 2
    holdout_class_list.extend(fruit_and_vegetables_list)  # 3
    holdout_class_list.extend(insects_list)  # 4
    holdout_class_list.extend(medium_mammals_list)  # 5
    holdout_class_list.extend(people_list)  # 6
    holdout_class_list.extend(reptiles_list)  # 7
    holdout_class_list.extend(small_mammals_list)  # 8
    holdout_class_list.extend(trees_list)  # 9

    runs_list = []
    for holdout_class in holdout_class_list:
        # for ratio in [0,3,6,9]:
        # for ratio in [1,2,4,5,7,8,10]:
        for ratio in range(11):
            # for ratio in [10]:
            for run_id in range(no_runs):
                outputs_file = os.path.join(local_result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, ratio), 'outputs_%s' % run_id)
                if not check_done(outputs_file):
                    script = "cluster_single_cifar100.py"
                    args = "%s %s %d %d" % (mode, holdout_class, ratio, run_id)
                    log = "cifar100_%s_%s_%d_%d" % (mode, holdout_class, ratio, run_id)
                    runs_list.append((script, args, log, True))

    return runs_list


def list_cifar100_holdout_val_test_runs(local_result_folder, no_runs):
    mode = 'holdout'
    #holdout_class_list = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
    #holdout_class_list = ['caterpillar', 'mushroom', 'porcupine', 'ray']

    #holdout_class_list = ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum', 'squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal']
    # holdout_class_list = ['boy', 'maple_tree', 'oak_tree', 'willow_tree', 'man', 'woman', 'pine_tree',
    #                      'apple', 'girl', 'orange', 'rose', 'cockroach', 'tulip', 'baby', 'palm_tree', 'poppy', 'pear', 'whale']

    aquatic_mammals_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale']  # 0
    fish_list = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']  # 1
    flower_list = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']  # 2
    fruit_and_vegetables_list = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']  # 3
    insects_list = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']  # 4
    medium_mammals_list = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']  # 5
    people_list = ['baby', 'boy', 'girl', 'man', 'woman']  # 6
    reptiles_list = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']  # 7
    small_mammals_list = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']  # 8
    trees_list = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']  # 9

    holdout_class_list = []
    holdout_class_list.extend(aquatic_mammals_list)  # 0
    holdout_class_list.extend(fish_list)  # 1
    holdout_class_list.extend(flower_list)  # 2
    holdout_class_list.extend(fruit_and_vegetables_list)  # 3
    holdout_class_list.extend(insects_list)  # 4
    holdout_class_list.extend(medium_mammals_list)  # 5
    holdout_class_list.extend(people_list)  # 6
    holdout_class_list.extend(reptiles_list)  # 7
    holdout_class_list.extend(small_mammals_list)  # 8
    holdout_class_list.extend(trees_list)  # 9

    runs_list = []
    for holdout_class in holdout_class_list:
        # for ratio in [0,3,6,9]:
        # for ratio in [1,2,4,5,7,8,10]:
        for ratio in range(11):
            # for ratio in [10]:
            for run_id in range(no_runs):
                outputs_file = os.path.join(local_result_folder, 'cifar100', 'cifa-r100-%s_%s_%d' % (mode, holdout_class, ratio), 'outputs_var_%s' % run_id)
                if not check_done(outputs_file):
                    script = "cluster_single_val_test_cifar100.py"
                    args = "%s %s %d %d" % (mode, holdout_class, ratio, run_id)
                    log = "cifar100_%s_%s_%d_%d" % (mode, holdout_class, ratio, run_id)
                    runs_list.append((script, args, log, True))

    return runs_list


def list_cifar100_holdout_val_train_more_runs(local_result_folder, no_runs):
    mode = 'holdout'

    #holdout_class_list = ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum', 'squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal']
    # holdout_class_list = ['boy', 'maple_tree', 'oak_tree', 'willow_tree', 'man', 'woman', 'pine_tree',
    #                      'apple', 'girl', 'orange', 'rose', 'cockroach', 'tulip', 'baby', 'palm_tree', 'poppy', 'pear', 'whale']

    aquatic_mammals_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale']  # 0
    fish_list = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']  # 1
    flower_list = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']  # 2
    fruit_and_vegetables_list = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']  # 3
    insects_list = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']  # 4
    medium_mammals_list = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']  # 5
    people_list = ['baby', 'boy', 'girl', 'man', 'woman']  # 6
    reptiles_list = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']  # 7
    small_mammals_list = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']  # 8
    trees_list = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']  # 9

    holdout_class_list = []
    holdout_class_list.extend(aquatic_mammals_list)  # 0
    holdout_class_list.extend(fish_list)  # 1
    holdout_class_list.extend(flower_list)  # 2
    holdout_class_list.extend(fruit_and_vegetables_list)  # 3
    holdout_class_list.extend(insects_list)  # 4
    holdout_class_list.extend(medium_mammals_list)  # 5
    holdout_class_list.extend(people_list)  # 6
    holdout_class_list.extend(reptiles_list)  # 7
    holdout_class_list.extend(small_mammals_list)  # 8
    holdout_class_list.extend(trees_list)  # 9

    runs_list = []
    for holdout_class in holdout_class_list:
        for retrain_mode in ['std_avg_conf']:
            # for retrain_mode in ['random', 'std_conf', 'avg_conf', 'std_avg_conf', 'single_conf']:
            # for retrain_mode in ['random']:
            # for ratio in [0,3,6,9]:
            # for ratio in [1,2,4,5,7,8,10]:
            # for ratio in range(11):
            for ratio in [0, 5, 10]:
                # for ratio in [10]:
                for val_ratio in [0, 1, 2, 5, 10]:
                    for run_id in range(no_runs):
                        outputs_file = os.path.join(local_result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, ratio), 'outputs_%d_%s_%d' % (run_id, retrain_mode, val_ratio))
                        if not check_done(outputs_file):
                            script = "cluster_single_train_more_cifar100.py"
                            args = "%s %s %d %d %d %s" % (mode, holdout_class, ratio, run_id, val_ratio, retrain_mode)
                            log = "cifar100_%s_%s_%d_%d_%s_%d" % (mode, holdout_class, ratio, run_id, retrain_mode, val_ratio)
                            runs_list.append((script, args, log, True))

    return runs_list


def list_cifar100_holdout_evaluate_runs(no_runs):
    mode = 'holdout'
    holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
    #holdout_classes = ['caterpillar', 'mushroom', 'porcupine', 'ray']

    runs_list = []
    for i in range(len(holdout_classes)):
        holdout_class = holdout_classes[i]
        for ratio in range(11):
            script = "cluster_single_evaluate_cifar100.py"
            args = "%s %s %d %d" % (mode, holdout_class, ratio, no_runs)
            log = "cifar100_evaluate_%s_%s_%d_%d" % (mode, holdout_class, ratio, no_runs)
            runs_list.append((script, args, log, False))

    return runs_list


def list_cifar100_holdout_analysis_runs(no_runs, script_name):
    mode = 'holdout'
    holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
    holdout_targets = [6, 6, 6, 6, 6, 4, 3, 5, 1]
    #holdout_classes = ['caterpillar', 'mushroom', 'porcupine', 'ray']
    #holdout_targets = [4,3,5,1]

    runs_list = []
    for i in range(len(holdout_classes)):
        holdout_class = holdout_classes[i]
        holdout_target = holdout_targets[i]
        for ratio in range(11):
            # for ratio in [10]:
            script = script_name
            args = "%s %s %d %d %d" % (mode, holdout_class, holdout_target, ratio, no_runs)
            log = "cifar100_analyze_%s_%s_%d_%d_%d" % (mode, holdout_class, holdout_target, ratio, no_runs)
            runs_list.append((script, args, log, False))

    return runs_list


def list_cifar100_holdout_generate_map_runs(local_result_folder, no_runs):
    mode = 'holdout'
    holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
    #holdout_classes = ['caterpillar', 'mushroom', 'porcupine', 'ray']

    runs_list = []
    for holdout_class in holdout_classes:
        # for ratio in [0,3,6,9]:
        # for ratio in [1,2,4,5,7,8,10]:
        for ratio in range(11):
            # for ratio in [10]:
            for run_id in range(no_runs):
                outputs_file = os.path.join(local_result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, ratio), 'maps_%s' % run_id)
                if not check_done(outputs_file):
                    script = "cluster_single_saliency_cifar100.py"
                    args = "%s %s %d %d" % (mode, holdout_class, ratio, run_id)
                    log = "cifar100_map_%s_%s_%d_%d" % (mode, holdout_class, ratio, run_id)
                    runs_list.append((script, args, log, True))

    return runs_list


def list_cifar10_runs(local_result_folder, no_runs):

    holdout_class = 0

    modes = ['holdout', 'holdout-dup', 'augmentation', 'augmentation-all']

    runs_list = []

    for mode in modes:
        # for ratio in [0,3,6,9]:
        for ratio in [1, 2, 4, 5, 7, 8, 10]:
            for run_id in range(no_runs):
                outputs_file = os.path.join(local_result_folder, 'cifar10', 'cifar10-%s_%d_%d' % (mode, holdout_class, ratio), 'outputs_%s' % run_id)
                if not os.path.isfile(outputs_file):
                    script = "cluster_single_cifar10.py"
                    args = "%s %d %d" % (mode, ratio, run_id)
                    log = "cifar100_%s_%d_%d_%d" % (mode, holdout_class, ratio, run_id)
                    runs_list.append((script, args, log, True))

    return runs_list


def list_amazon_holdout_runs(local_result_folder, no_runs):
    mode = 'holdout'

    Education_Reference = ["Languages", "Maps & Atlases", "Test Preparation", "Dictionaries", "Religion"]  # 0
    Business_Office = ["Office Suites", "Document Management", "Training", "Word Processing", "Contact Management"]  # 1
    Children_s = ["Early Learning", "Games", "Math", "Reading & Language", "Art & Creativity"]  # 2
    Utilities = ["Backup", "PC Maintenance", "Drivers & Driver Recovery", "Internet Utilities", "Screen Savers"]  # 3
    Design_Illustration = ["Animation & 3D", "Training", "CAD", "Illustration"]  # 4
    Accounting_Finance = ["Business Accounting", "Personal Finance", "Check Printing", "Point of Sale", "Payroll"]  # 5
    Video = ["Video Editing", "DVD Viewing & Burning", "Compositing & Effects", "Encoding"]  # 6
    Music = ["Instrument Instruction", "Music Notation", "CD Burning & Labeling", "MP3 Editing & Effects"]  # 7
    Programming_Web_Development = ["Training & Tutorials", "Programming Languages", "Database", "Development Utilities", "Web Design"]  # 8
    Networking_Servers = ["Security", "Firewalls", "Servers", "Network Management", "Virtual Private Networks"]  # 9

    holdout_class_list = []
    holdout_class_list.extend(Education_Reference)
    holdout_class_list.extend(Business_Office)
    holdout_class_list.extend(Children_s)
    holdout_class_list.extend(Utilities)
    holdout_class_list.extend(Design_Illustration)
    holdout_class_list.extend(Accounting_Finance)
    holdout_class_list.extend(Video)
    holdout_class_list.extend(Music)
    holdout_class_list.extend(Programming_Web_Development)
    holdout_class_list.extend(Networking_Servers)

    runs_list = []
    for holdout_class in holdout_class_list:
        # for ratio in [0,3,6,9]:
        # for ratio in [1,2,4,5,7,8,10]:
        for ratio in range(11):
            # for ratio in [10]:
            for run_id in range(no_runs):
                #outputs_file = os.path.join(local_result_folder, 'amazon', 'amazon-%s_%s_%d' % (mode, holdout_class.replace(' ', '_'), ratio), 'outputs_%s' % run_id)
                outputs_file = os.path.join(local_result_folder, 'amazon', 'amazon-%s_%s_%d' % (mode, holdout_class, ratio), 'outputs_%s' % run_id)
                if not check_done(outputs_file):
                    script = "cluster_single_amazon.py"
                    args = "%s %s %d %d" % (mode, holdout_class, ratio, run_id)
                    log = "amazon_%s_%s_%d_%d" % (mode, holdout_class.replace(' ', '_'), ratio, run_id)
                    runs_list.append((script, args, log, True))

    return runs_list


parser = argparse.ArgumentParser(description='Schedule experiment runs on the clusters with holdout CIFAR-100 and Resnet18')
parser.add_argument('local_teamdrive_folder', help='local teamdrive folder')
parser.add_argument('job', choices=['val_test', 'retrain', 'train'], help='which job to run')

args = parser.parse_args()

local_teamdrive_folder = args.local_teamdrive_folder

NO_RUNS = 25

# if args.job == 'val_test':
#    runs_list = list_cifar100_holdout_val_test_runs(os.path.join(local_teamdrive_folder, 'result'), NO_RUNS)
# elif args.job == 'retrain':
#    runs_list = list_cifar100_holdout_val_train_more_runs(os.path.join(local_teamdrive_folder, 'result'), NO_RUNS)

if args.job == 'train':
    runs_list = list_amazon_holdout_runs(os.path.join(local_teamdrive_folder, 'result'), NO_RUNS)
elif args.job == 'val_test':
    runs_list = []
elif args.job == 'retrain':
    runs_list = []

#runs_list = list_cifar10_runs(os.path.join(local_teamdrive_folder, 'result'), NO_RUNS)

#runs_list = list_cifar100_holdout_runs(os.path.join(local_teamdrive_folder, 'result'), NO_RUNS)

#runs_list = list_cifar100_holdout_evaluate_runs(NO_RUNS)

#runs_list = list_cifar100_holdout_analysis_runs(NO_RUNS, "cluster_single_detail_analysis_cifar100.py")

#runs_list = list_cifar100_holdout_generate_map_runs(os.path.join(local_teamdrive_folder, 'result'), NO_RUNS)

#runs_list = list_cifar100_holdout_analysis_runs(NO_RUNS, "cluster_single_detail_analysis_cifar100_html.py")

#MAX_NO_JOBS = 64
MAX_NO_JOBS = 16
#MAX_NO_JOBS = 32

MIN_NO_RUNS_PER_JOB = 10

#NO_RUNS_PER_JOB = 1
NO_RUNS_PER_JOB = math.ceil(len(runs_list)/MAX_NO_JOBS)
if NO_RUNS_PER_JOB < MIN_NO_RUNS_PER_JOB:
    NO_RUNS_PER_JOB = MIN_NO_RUNS_PER_JOB
no_jobs = math.ceil(len(runs_list)/NO_RUNS_PER_JOB)
for job_id in range(no_jobs):
    start = job_id * NO_RUNS_PER_JOB
    end = min((job_id + 1) * NO_RUNS_PER_JOB, len(runs_list))
    runs = runs_list[start:end]

    schedule_train_job("data", "result", 'modelvar-train-rn18-holdout', runs, runs[0][3])
